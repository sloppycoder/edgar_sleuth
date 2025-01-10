import logging
from datetime import datetime

from .datastore import execute_query, initialize_search_phrases
from .llm.algo import (
    gather_chunk_distances,
    most_relevant_chunks,
    relevance_by_distance,
)
from .llm.embedding import batch_embedding
from .llm.extraction import ask_model, extract_json_from_response

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


def create_search_phrase_embeddings(
    table_name: str, model: str, tag: str, dimension: int
) -> None:
    embeddings = batch_embedding(
        chunks=TRUSTEE_COMP_SEARCH_PHRASES,
        model=model,
        task_type="RETRIEVAL_QUERY",
        dimension=dimension,
    )
    data = list(zip(TRUSTEE_COMP_SEARCH_PHRASES, embeddings))
    initialize_search_phrases(table_name=table_name, data=data, tags=[tag])


def get_relevant_chunks_with_distances(
    cik: str,
    accession_number: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    search_phrase_tag: str,
    embedding_tag: str,
):
    # limit 12 is like top_k =3 for 4 search phrases
    result = execute_query(
        f"""
        SELECT
            cik, accession_number, phrase, chunk_num,
            embedding <=> phrase_embedding as distance
        FROM
            {search_phrase_table_name} phrases,
            {embedding_table_name} docs
        WHERE
            cik = %s AND accession_number = %s
            AND %s = ANY(phrases.tags) AND %s = ANY(docs.tags)
            ORDER BY
                embedding <=> phrase_embedding
            limit 12;
    """,
        (cik, accession_number, search_phrase_tag, embedding_tag),
    )
    return result


def get_text_by_chunk_num(
    text_table_name: str,
    cik: str,
    accession_number: str,
    chunk_nums: list[int],
    tag: str,
) -> str:
    result = execute_query(
        f"""
        SELECT
            STRING_AGG(chunk_text, '\n' ORDER BY chunk_num) as relevant_text
        FROM
            {text_table_name}
        WHERE
            cik = %s AND accession_number = %s
            AND %s = ANY(tags) AND chunk_num = ANY(%s)
    """,
        (cik, accession_number, tag, chunk_nums),
    )
    if result:
        return result[0]["relevant_text"]
    else:
        return ""


def find_relevant_text(
    cik: str,
    accession_number: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    tag: str,
    search_phrase_tag: str,
) -> str:
    relevance_result = get_relevant_chunks_with_distances(
        cik=cik,
        accession_number=accession_number,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        search_phrase_tag=search_phrase_tag,
        embedding_tag=tag,
    )

    chunk_distances = gather_chunk_distances(relevance_result)
    relevance_scores = relevance_by_distance(chunk_distances)
    selected_chunks = [int(s) for s in most_relevant_chunks(relevance_scores)]
    logger.debug(f"Selected chunks: {selected_chunks}")

    relevant_text = get_text_by_chunk_num(
        cik=cik,
        accession_number=accession_number,
        chunk_nums=selected_chunks,
        text_table_name=text_table_name,
        tag=tag,
    )
    if relevant_text and len(relevant_text) > 100:
        return relevant_text
    else:
        return ""


def ask_model_about_trustee_comp(model: str, relevant_text: str):
    start_t = datetime.now()
    prompt = TRUSTEE_COMP_PROMPT.replace("{SEC_FILING_SNIPPET}", relevant_text)
    response = ask_model(model, prompt)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"ask {model} with prompt of {len(prompt)} took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )

    if response:
        comp_info = extract_json_from_response(response)
        return response, comp_info
    return None, None


def extract_trustee_comp(
    cik: str,
    accession_number: str,
    search_phrase_table_name: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_tag: str,
    tag: str,
    model: str,
) -> tuple[str | None, dict | None]:
    # the extractino process has 4 steps
    # step 1: chunk the filing
    # step 2: get embedding
    # the above 2 steps are outside this function

    # step 3: using search phrases to run vector search
    # use scoring alborithm to determine the most relevant text chunks
    relevant_text = find_relevant_text(
        cik=cik,
        accession_number=accession_number,
        text_table_name=text_table_name,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        tag=tag,
        search_phrase_tag=search_phrase_tag,
    )
    if not relevant_text or len(relevant_text) < 100:
        logger.info(
            f"No relevant text found for {cik},{accession_number} with tags {search_phrase_tag} and {tag}"  # noqa E501
        )

    # step 4: send the relevant text to the LLM model with designed prompt
    response, comp_info = ask_model_about_trustee_comp(model, relevant_text)
    if response and comp_info:
        return response, comp_info

    return None, None
