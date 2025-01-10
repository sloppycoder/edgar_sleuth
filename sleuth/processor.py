import logging
from logging.handlers import QueueHandler
from typing import Any

from .edgar import SECFiling
from .llm.embedding import save_filing_embeddings
from .splitter import chunk_filing
from .trustee import extract_trustee_comp

logger = logging.getLogger(__name__)


def log_n_print(message):
    print(message)
    logger.info(message)


def process_filing(
    actions: list[str],
    search_tag: str,
    dimension: int,
    cik: str,
    accession_number: str,
    tags: list[str],
    model: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    form_type: str,
) -> dict[str, Any]:
    ret_val = {}

    key = f"Filing({cik},{accession_number})"
    log_n_print(
        f"Processing {key} for {actions} with search_tag={search_tag}, tags={tags}"
    )
    ret_val[key] = key

    if "chunk" in actions:
        filing = SECFiling(cik=cik, accession_number=accession_number)
        n_chunks, _ = chunk_filing(
            filing=filing,
            form_type=form_type,
            tags=tags,
            table_name=text_table_name,
        )
        if n_chunks > 1:
            log_n_print(f"{key} {form_type} splitted into {n_chunks} chunks")
        else:
            log_n_print(f"Error when splitting {key} {form_type}")

        ret_val["n_chunks"] = n_chunks

    if "embedding" in actions:
        n_embeddings = save_filing_embeddings(
            text_table_name=text_table_name,
            cik=cik,
            accession_number=accession_number,
            tag=tags[0],
            embedding_table_name=embedding_table_name,
            dimension=dimension,
        )
        if n_embeddings > 1:
            log_n_print(f"Saved {n_embeddings} embeddings for {key} {form_type}")
        else:
            log_n_print(f"Error when get embeddings for {key} {form_type}")

        ret_val["n_embeddings"] = n_embeddings

    if "extract" in actions:
        response, comp_info = extract_trustee_comp(
            cik=cik,
            accession_number=accession_number,
            text_table_name=text_table_name,
            embedding_table_name=embedding_table_name,
            search_phrase_table_name=search_phrase_table_name,
            tag=tags[0],
            search_phrase_tag=search_tag,
            model=model,
        )
        logger.debug(f"{model} response:{response}")
        n_trustees = len(comp_info["trustees"]) if comp_info else 0
        log_n_print(f"Extracted {n_trustees} from {key}")

        ret_val["response"] = response
        ret_val["comp_info"] = comp_info

    return ret_val


def process_filing_wrapper(args: dict):
    # wrapper for multiprocessing
    # init logging for each worker to use QueueHandler that sends the logs
    # to the main process
    try:
        process_filing(**args)
    except Exception as e:
        logger.error(f"Error {str(e)} in process_filing: {args["filing"]}")


def init_worker(logging_q, log_level=logging.DEBUG):
    # remove exsiting handlers with QueueHandler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("sleuth").setLevel(log_level)
    logging.getLogger("sleuth.datastore").setLevel(logging.INFO)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(logging_q))
