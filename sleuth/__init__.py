import logging
from datetime import datetime
from logging.handlers import QueueHandler

from .datastore import get_chunks, save_chunks
from .edgar import SECFiling
from .llm.embedding import GEMINI_EMBEDDING_MODEL, batch_embedding
from .splitter import chunk_text, trim_html_content
from .trustee import extract_trustee_comp

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def chunk_filing(
    filing: SECFiling,
    form_type: str,
    method: str = "spacy",
    tags: list[str] = [],
    table_name: str = "",  # leave empty if dryrun
) -> int:
    logger.debug(f"chunk_filing form {form_type} of {filing}")

    if filing:
        filing_path, filing_content = filing.get_doc_content(form_type, max_items=1)[0]

        if not filing_path.endswith(".html") and not filing_path.endswith(".htm"):
            logger.info(f"{filing_path} is not html file, skipping...")
            return 0

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
                    tags=tags,
                    create_table=True,
                )
            return len(chunks)

    return 0


def get_embeddings(
    text_table_name: str,
    cik: str,
    accession_number: str,
    dimension: int,
    tag: str,
    embedding_table_name: str,
    model: str = GEMINI_EMBEDDING_MODEL,
) -> int:
    text_chunks_records = get_chunks(
        cik=cik,
        accession_number=accession_number,
        table_name=text_table_name,
        tag=tag,
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
                tags=[tag],
                create_table=True,
            )
        return len(embeddings)

    return 0


def log_n_print(message):
    print(message)
    logger.info(message)


def process_filing(
    actions: list[str],
    search_tag: str,
    dimension: int,
    filing: SECFiling,
    tags: list[str],
    model: str,
    text_table_name: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    form_type: str,
):
    log_n_print(
        f"Processing {filing} for {actions} with search_tag={search_tag}, tags={tags}"
    )

    if "chunk" in actions:
        n_chunks = chunk_filing(
            filing=filing,
            form_type=form_type,
            tags=tags,
            table_name=text_table_name,
        )
        if n_chunks > 1:
            log_n_print(f"{filing} {form_type} splitted into {n_chunks} chunks")
        else:
            log_n_print(f"Error when splitting {filing} {form_type}")

    if "embedding" in actions:
        n_chunks = get_embeddings(
            text_table_name=text_table_name,
            cik=filing.cik,
            accession_number=filing.accession_number,
            tag=tags[0],
            embedding_table_name=embedding_table_name,
            dimension=dimension,
        )
        if n_chunks > 1:
            log_n_print(f"Saved {n_chunks} embeddings for {filing} {form_type}")
        else:
            log_n_print(f"Error when get embeddings for {filing} {form_type}")

    if "extract" in actions:
        response, comp_info = extract_trustee_comp(
            cik=filing.cik,
            accession_number=filing.accession_number,
            text_table_name=text_table_name,
            embedding_table_name=embedding_table_name,
            search_phrase_table_name=search_phrase_table_name,
            tag=tags[0],
            search_phrase_tag=search_tag,
            model=model,
        )
        logger.debug(f"{model} response:{response}")
        n_trustees = len(comp_info["trustees"]) if comp_info else 0
        log_n_print(f"Extracted {n_trustees} from {filing}")


def process_filing_wrapper(args: dict):
    # wrapper for multiprocessing
    # init logging for each worker to use QueueHandler that sends the logs
    # to the main process
    try:
        process_filing(**args)
    except Exception as e:
        logger.error(f"Error {str(e)} in process_filing: {args["filing"]}")


def init_worker(logging_q):
    # remove exsiting handlers with QueueHandler
    root_logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    root_logger.addHandler(QueueHandler(logging_q))
