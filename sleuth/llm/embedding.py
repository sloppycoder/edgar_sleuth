import logging
from datetime import datetime

import httpx
import tiktoken
from google.api_core.exceptions import GoogleAPICallError, ServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from ..datastore import get_chunks, save_chunks
from .util import init_vertaxai, openai_client

# models used for embeddings
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
GEMINI_EMBEDDING_MODEL = "text-embedding-005"

logger = logging.getLogger(__name__)


def save_filing_embeddings(
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


def batch_embedding(
    chunks: list[str],
    model: str,
    dimension: int,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Generates embeddings for a list of text chunks using either OpenAI or
    Gemini embedding model.

    Args:
        chunks (list[str]): A list of text chunks to process

    Returns:
        list[list[float]]: A list of embeddings (one embedding per chunk)
    """
    if model == OPENAI_EMBEDDING_MODEL:
        max_tokens_per_request, max_chunks_per_request = 8191, 99999  # no limit
    elif model == GEMINI_EMBEDDING_MODEL:
        max_tokens_per_request, max_chunks_per_request = 12000, 250
    else:
        raise ValueError(f"Unsupported embedding model {model}")

    # tiktoken does not support Gemini model
    # use OpenAI as stand-in.
    # since token limit is an OpenAI issue anyways.
    encoding = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)

    embeddings = []

    # Split chunks into smaller batches based on token limit
    current_batch: list[str] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk))

        if (
            len(current_batch) >= max_chunks_per_request
            or (current_tokens + chunk_tokens) > max_tokens_per_request
        ):
            # Process the current batch before adding the new chunk
            result = _call_embedding_api(
                content=current_batch,
                model=model,
                task_type=task_type,
                dimension=dimension,
            )
            embeddings.extend(result)
            current_batch = []
            current_tokens = 0

        # Add the chunk to the current batch
        current_batch.append(chunk)
        current_tokens += chunk_tokens

    # Process the final batch if it exists
    if current_batch:
        result = _call_embedding_api(
            content=current_batch,
            model=model,
            task_type=task_type,
            dimension=dimension,
        )
        embeddings.extend(result)

    return embeddings


def _call_embedding_api(
    content: list[str],
    model: str,
    task_type: str,
    dimension: int,
) -> list[list[float]]:
    if model == OPENAI_EMBEDDING_MODEL:
        return _call_openai_embedding_api(
            content,
            model=model,
        )
    elif model == GEMINI_EMBEDDING_MODEL:
        return _call_gemini_embedding_api(
            content,
            model=model,
            task_type=task_type,
            dimensionality=dimension,
        )
    else:
        raise ValueError(f"Unsupported embedding model {model}")


class RetriableServerError(Exception):
    """
    Indicate server side error when calling an API
    This can be used to trigger a retry.
    """

    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RetriableServerError),
)
def _call_openai_embedding_api(input_: list[str], model: str) -> list[list[float]]:
    try:
        client = openai_client()
        response = client.embeddings.create(input=input_, model=model)
        return [item.embedding for item in response.data]
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            raise RetriableServerError(e)
        else:
            raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RetriableServerError),
)
def _call_gemini_embedding_api(
    content: list[str], model: str, task_type: str, dimensionality: int
) -> list[list[float]]:
    try:
        init_vertaxai()
        embedding_model = TextEmbeddingModel.from_pretrained(model)
        inputs = [TextEmbeddingInput(text, task_type=task_type) for text in content]
        embeddings = embedding_model.get_embeddings(
            texts=inputs,  # pyright: ignore
            auto_truncate=True,
            output_dimensionality=dimensionality,
        )
        return [e.values for e in embeddings]
    except GoogleAPICallError as e:
        if isinstance(e, ServerError):
            raise RetriableServerError(e)
        else:
            raise
