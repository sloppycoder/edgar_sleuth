import logging
from datetime import datetime
from logging.handlers import QueueHandler
from typing import Any

from .datastore import (
    DatabaseException,
    execute_insertmany,
    execute_query,
    get_chunks,
    save_chunks,
)
from .edgar import SECFiling
from .llm.embedding import GEMINI_EMBEDDING_MODEL, batch_embedding
from .splitter import chunk_text, trim_html_content
from .trustee import extract_trustee_comp

logger = logging.getLogger(__name__)


def log_n_print(message):
    print(message)
    logger.info(message)


def save_filing_embeddings(
    text_table_name: str,
    cik: str,
    accession_number: str,
    dimension: int,
    embedding_table_name: str = "",  # leave empty to skip saving to database
    model: str = GEMINI_EMBEDDING_MODEL,
) -> int | None:
    logger.debug(
        f"save_filing_embeddings for {cik},{accession_number} with dimension {dimension}, model={model}"  # noqa E501
    )

    # check if embeddings already exist
    if embedding_table_name:
        try:
            query = f"""
                SELECT COUNT(*) AS COUNT FROM {embedding_table_name}
                WHERE cik = %s AND accession_number = %s
            """
            result = execute_query(query, (cik, accession_number))
            if result and result[0]["count"] > 0:
                logger.debug(
                    f"{cik} {accession_number} already has embeddings, skipping calling embedding API"  # noqa E501
                )
                return result[0]["count"]
        except DatabaseException as e:
            if "does not exist" not in str(e):
                raise e

    text_chunks_records = get_chunks(
        cik=cik,
        accession_number=accession_number,
        table_name=text_table_name,
    )
    logger.debug(
        f"Retrieved {len(text_chunks_records)} text chunks for {cik} {accession_number}"
    )
    chunks = [record["chunk_text"] for record in text_chunks_records]

    start_t = datetime.now()
    embeddings = batch_embedding(chunks, model=model, dimension=dimension)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"batch_embedding of {len(chunks)} chunks of text with {model} took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )

    if len(embeddings) > 1:
        if embedding_table_name:
            logger.debug(f"Saving {len(chunks)} embeddings to {embedding_table_name}")
            save_chunks(
                cik=cik,
                accession_number=accession_number,
                chunks=embeddings,
                table_name=embedding_table_name,
                create_table=True,
            )
        return len(embeddings)

    return None


def chunk_filing(
    filing: SECFiling,
    form_type: str,
    method: str = "spacy",
    table_name: str = "",  # leave empty if dryrun
) -> tuple[int, list[str]] | tuple[None, None]:
    logger.debug(f"chunk_filing form {form_type} of {filing}")

    if filing:
        filing_path, filing_content = filing.get_doc_content(form_type, max_items=1)[0]

        if not filing_path.endswith(".html") and not filing_path.endswith(".htm"):
            logger.info(f"{filing_path} is not html file, skipping...")
            return None, None

        # check if the filing is already chunk
        if table_name:
            try:
                existing_chunks = get_chunks(
                    cik=filing.cik,
                    accession_number=filing.accession_number,
                    table_name=table_name,
                )
                if existing_chunks:
                    logger.info(f"{filing.cik} {filing.accession_number} already chunked")
                    return len(existing_chunks), [
                        record["chunk_text"] for record in existing_chunks
                    ]
            except DatabaseException as e:
                if "does not exist" not in str(e):
                    raise e

        trimmed_html = trim_html_content(filing_content)
        logger.debug(f"Trimmed HTML content size {len(trimmed_html)}")

        start_t = datetime.now()
        chunks = chunk_text(trimmed_html, method=method)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"chunking with {len(filing_content)} of text with {method} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )

        if len(chunks) > 1:
            if table_name:
                logger.debug(f"Saving {len(chunks)} text chunks to {table_name}")
                save_chunks(
                    cik=filing.cik,
                    accession_number=filing.accession_number,
                    chunks=chunks,
                    table_name=table_name,
                    create_table=True,
                )
            return len(chunks), chunks

    return None, None


def gather_extractin_result(
    idx_table_name: str,
    extraction_result_table_name: str,
    idx_tag: str,
    result_tag: str,
) -> list[dict[str, Any]]:
    query = f"""
        SELECT DISTINCT
            idx.cik,
            company_name,
            form_type,
            TO_CHAR(date_filed, 'YYYY-MM-DD') as date_filed,
            idx_filename as filename,
            idx.accession_number,
            res.selected_chunks as chunks_used,
            res.selected_text as relevant_text,
            res.n_trustee as num_trustees,
            res.response as trustees_comp
        FROM {idx_table_name} idx
        LEFT JOIN {extraction_result_table_name} res
            ON res.cik = idx.cik
            AND res.accession_number = idx.accession_number
            AND %s = ANY(res.tags)
        WHERE %s = ANY(idx.tags)
        LIMIT 10000
    """

    return execute_query(query, (result_tag, idx_tag))


def process_filing(
    action: str,
    tables_map: dict[str, str],
    cik: str,
    accession_number: str,
    idx_tag: str,
    search_tag: str,
    result_tag: str,
    model: str,
    dimension: int,
    form_type: str,
) -> bool:
    key = f"Filing({cik},{accession_number})"
    log_n_print(f"Processing {key} for {action} with idx_tag={idx_tag}")

    if action == "chunk":
        filing = SECFiling(cik=cik, accession_number=accession_number)
        n_chunks, _ = chunk_filing(
            filing=filing,
            form_type=form_type,
            table_name=tables_map["text"],
        )
        if n_chunks:
            log_n_print(f"{key} {form_type} splitted into {n_chunks} chunks")
            return True
        else:
            log_n_print(f"Error when splitting {key} {form_type}")
            return False

    elif action == "embedding":
        n_embeddings = save_filing_embeddings(
            text_table_name=tables_map["text"],
            cik=cik,
            accession_number=accession_number,
            embedding_table_name=tables_map["embedding"],
            dimension=dimension,
        )
        if n_embeddings:
            log_n_print(f"Saved {n_embeddings} embeddings for {key} {form_type}")
            return True
        else:
            log_n_print(f"Error when get embeddings for {key} {form_type}")
            return False

    if action == "extract":
        extraction_result = extract_trustee_comp(
            cik=cik,
            accession_number=accession_number,
            text_table_name=tables_map["text"],
            embedding_table_name=tables_map["embedding"],
            search_phrase_table_name=tables_map["search"],
            search_phrase_tag=search_tag,
            model=model,
        )

        if extraction_result:
            # logger.debug(f"{model} response:{response}")
            log_n_print(f"Extracted {extraction_result["n_trustee"]} from {key}")

            extraction_result["tags"] = [result_tag]
            result_saved = execute_insertmany(
                table_name=tables_map["result"],
                data=[extraction_result],
                create_table=True,
            )

            if result_saved:
                return True

        log_n_print(f"Error when saving {key} trustee comp result")
        return False

    else:
        log_n_print(f"Unknown action {action}")
        return False


def process_filing_wrapper(args: dict):
    # wrapper for multiprocessing
    # init logging for each worker to use QueueHandler that sends the logs
    # to the main process
    try:
        process_filing(**args)
    except Exception as e:
        logger.error(
            f"Error {str(e)} in process_filing: Filing({args["cik"]},{args["accession_number"]})"  # noqa E501
        )


def init_worker(logging_q, log_level=logging.DEBUG):
    # remove exsiting handlers with QueueHandler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("sleuth").setLevel(log_level)
    logging.getLogger("sleuth.datastore").setLevel(logging.INFO)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(logging_q))
