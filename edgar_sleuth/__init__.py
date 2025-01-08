import logging
from datetime import datetime

from datastore import get_chunks, save_chunks
from edgar import SECFiling
from llm.embedding import GEMINI_EMBEDDING_MODEL, batch_embedding
from splitter import chunk_text

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def chunk_filing(
    filing: SECFiling,
    form_type: str,
    dryrun: bool = False,
    method: str = "spacy",
    tags: list[str] = [],
    table_name: str = "",
) -> int:
    if not table_name and not dryrun:
        raise ValueError("table_name is required if not in dryrun mode")

    logging.debug(f"chunk_filing form {form_type} of {filing}")

    if filing:
        _, filing_content = filing.get_doc_content(form_type, max_items=1)[0]

        start_t = datetime.now()
        chunks = chunk_text(filing_content, method=method)
        elapsed_t = datetime.now() - start_t
        logger.debug(
            f"chunking with {len(filing_content)} of text with {method} took {elapsed_t.total_seconds()} seconds"  # noqa E501
        )

        if len(chunks) > 1:
            if not dryrun:
                logger.debug(f"Saving {len(chunks)} text chunks to {table_name}")
                save_chunks(
                    cik="1035018",
                    accession_number="0001193125-20-000327",
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
    tags: list[str] = [],
    model: str = GEMINI_EMBEDDING_MODEL,
    dryrun: bool = False,
    embedding_table_name: str = "",
) -> int:
    if not embedding_table_name and not dryrun:
        raise ValueError("embedding_table_name is required if not in dryrun mode")

    text_chunks_records = get_chunks(
        cik=cik,
        accession_number=accession_number,
        table_name=text_table_name,
    )
    chunks = [record["chunk_text"] for record in text_chunks_records]
    start_t = datetime.now()
    embeddings = batch_embedding(chunks, model=model)
    elapsed_t = datetime.now() - start_t
    logger.debug(
        f"batch_embedding of {len(chunks)} chunks of text with {model} took {elapsed_t.total_seconds()} seconds"  # noqa E501
    )

    if len(embeddings) > 1:
        if not dryrun:
            table_name = "filing_chunks_embeddings"
            logger.debug(f"Saving {len(chunks)} embeddings to {table_name}")
            save_chunks(
                cik=cik,
                accession_number=accession_number,
                chunks=embeddings,
                table_name=table_name,
                tags=tags,
                create_table=True,
                dimension=len(embeddings[0]),
            )
        return len(embeddings)

    return 0
