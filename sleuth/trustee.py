import json
import logging
from datetime import datetime
from typing import Any

from .datastore import (
    DatabaseException,
    execute_insertmany,
    execute_query,
    get_chunks,
    relevant_chunks_with_distances,
)
from .llm.algo import (
    gather_chunk_distances,
    relevance_by_appearance,
    relevance_by_distance,
    top_3_chunks,
    top_adjacent_chunks,
)
from .llm.embedding import batch_embedding
from .llm.extraction import ask_model, remove_md_json_wrapper

logger = logging.getLogger(__name__)

TRUSTEE_COMP_SEARCH_PHRASES = [
    "Trustee Compensation Structure and Amount",
    "Independent Director or Trustee Compensation Table",
    "Board Director or Intereed Person Compensation Details with Amount",
    "Interested Person Compensation Remuneration Detailed Amount",
]


TRUSTEE_COMP_PROMPT = """
You are tasked with extracting compensation information for Trustees from a snippet
of an SEC filing 485BPOS. Here is the snippet you need to analyze:

<sec_filing_snippet>
{SEC_FILING_SNIPPET}
</sec_filing_snippet>

Your task is to extract the following information:
1. Determine if compensation information for Trustees is present in the snippet.
2. If present, extract the compensation details for each Trustee, including their name, job title, fund compensation, fund group compensation, and deferred compensation.
3. Note any additional types of compensation mentioned in the document.

Follow these steps to analyze the snippet:
1. Carefully read through the entire snippet.
2. Look for a table or section that contains compensation information for Trustees, Board Members, Board Directors, or Interested Persons.
3. If you find such information, extract the relevant details for each Trustee.
4. Pay attention to any footnotes or additional explanations related to the compensation.

Structure your output as follows:
1. A boolean field indicating whether compensation information is present in the snippet.
2. A list of Trustees with their compensation details.
3. A notes field for any additional information or explanations.

If the compensation information is not present in the snippet:
1. Set the boolean field to false.
2. Leave the list of Trustees empty.
3. In the notes field, explain that the compensation information was not found in the given snippet.

If you find any additional relevant information or need to provide explanations about your analysis,
include them in the notes field.

Provide your output in JSON format, as showsn in example below

{
 "compensation_info_present": true/false,
 "trustees": [
  {
   "year": "Year of Compensation",
   "name": "name of the trustee or N/A",
   "job_title": "the job title of the person who is a trustee. e.g. Commitee Chairperson",
   "fund_compensation": "Amount or N/A",
   "fund_group_compensation": "Amount or N/A",
   "deferred_compensation": "Amount or N/A",
   "other_compensation": {
    "type": "Amount"
   }
  }
 ],
 "notes": "Any additional information or explanations"
}
Please remove the leading $ sign and comma from compensation Amount.
"""  # noqa: E501


def delete_search_pharses(table_name: str, search_tag: str) -> None:
    try:
        execute_query(f"DELETE FROM {table_name} WHERE tags = %s", ([search_tag],))
    except DatabaseException as e:
        if "does not exist" not in str(e):
            raise e


def create_search_phrase_embeddings(
    table_name: str,
    phrases: list[str],
    model: str,
    search_tag: str,
    dimension: int,
) -> None:
    embeddings = batch_embedding(
        chunks=phrases,
        model=model,
        task_type="RETRIEVAL_QUERY",
        dimension=dimension,
    )
    data = [
        {"phrase": phrase, "phrase_embedding": embedding}
        for phrase, embedding in zip(phrases, embeddings)
    ]
    for item in data:
        item["tags"] = [search_tag]

    execute_insertmany(table_name=table_name, data=data, create_table=True)
    logger.info(
        f"Initialized {len(data)} search phrases in {table_name} with size {dimension}"
    )


def extract_trustee_comp(
    cik: str,
    accession_number: str,
    search_phrase_table_name: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_tag: str,
    model: str,
) -> dict[str, Any] | None:
    # the extractino process has 4 steps
    # step 1: chunk the filing
    # step 2: get embedding
    # the above 2 steps are outside this function

    # step 3: using search phrases to run vector search
    # use scoring alborithm to determine the most relevant text chunks
    chunks_tried = []
    result = {
        "cik": cik,
        "accession_number": accession_number,
        "model": model,
        "response": "",
        "n_trustee": 0,
        "selected_chunks": [],
        "selected_text": "",
    }

    for method in ["distance", "appearance", "distance-any3", "appearance-any3"]:
        relevant_chunks, relevant_text = _find_relevant_text(
            cik=cik,
            accession_number=accession_number,
            text_table_name=text_table_name,
            embedding_table_name=embedding_table_name,
            search_phrase_table_name=search_phrase_table_name,
            search_phrase_tag=search_phrase_tag,
            method=method,
        )

        if (
            relevant_chunks in chunks_tried
            or not relevant_text
            or len(relevant_text) < 100
        ):  # noqa E501
            continue

        chunks_tried.append(relevant_chunks)
        result["selected_chunks"] = relevant_chunks
        result["selected_text"] = relevant_text

        # step 4: send the relevant text to the LLM model with designed prompt
        response = _ask_model_about_trustee_comp(model, relevant_text)
        if response:
            try:
                comp_info = json.loads(response)
                n_trustee = len(comp_info["trustees"])

                if n_trustee > 1:
                    result["n_trustee"] = n_trustee
                    result["response"] = response
                    return result
            except json.JSONDecodeError:
                pass

    if len(chunks_tried) > 0:
        logger.info(
            f"unable to extract trustee info for {cik},{accession_number} with tags {search_phrase_tag}"  # noqa E501
        )
        return result

    return None


def _find_relevant_text(
    cik: str,
    accession_number: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    search_phrase_tag: str,
    method: str,
) -> tuple[list[int], str]:
    relevance_result = relevant_chunks_with_distances(
        cik=cik,
        accession_number=accession_number,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        search_phrase_tag=search_phrase_tag,
        limit=20,
    )

    if not relevance_result:
        return [], ""

    chunk_distances = gather_chunk_distances(relevance_result)
    if method == "distance":
        relevance_scores = relevance_by_distance(chunk_distances)
        selected_chunks = [int(s) for s in top_adjacent_chunks(relevance_scores)]
    elif method == "appearance":
        relevance_scores = relevance_by_appearance(chunk_distances)
        selected_chunks = [int(s) for s in top_adjacent_chunks(relevance_scores)]
    elif method == "distance-any3":
        relevance_scores = relevance_by_distance(chunk_distances)
        selected_chunks = [int(s) for s in top_3_chunks(relevance_scores)]
    elif method == "appearance-any3":
        relevance_scores = relevance_by_appearance(chunk_distances)
        selected_chunks = [int(s) for s in top_3_chunks(relevance_scores)]
    else:
        return [], ""

    logger.debug(f"Selected chunks by {method}: {selected_chunks}")

    results = get_chunks(
        table_name=text_table_name,
        cik=cik,
        accession_number=accession_number,
        chunk_nums=selected_chunks,
    )

    if results and len(results) == len(selected_chunks):
        return selected_chunks, "\n".join([row["chunk_text"] for row in results])
    else:
        return [], ""


def _ask_model_about_trustee_comp(model: str, relevant_text: str) -> str | None:
    start_t = datetime.now()
    prompt = TRUSTEE_COMP_PROMPT.replace("{SEC_FILING_SNIPPET}", relevant_text)
    response = ask_model(model, prompt)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )

    if response:
        return remove_md_json_wrapper(response)

    return None
