# File: app/vectorstore/embedder.py
# Purpose: Generation of text embeddings using OpenAI or local SentenceTransformers.
# === File: app/vectorstore/embedder.py ===
import asyncio
from typing import Any, Protocol, cast

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from app.config import get_settings
from app.logging_config.setup import get_logger

logger = get_logger(__name__)


# 1. Define a strict Protocol for the untyped SentenceTransformer
class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str] | str) -> Any: ...


_local_model: SentenceEncoder | None = None


def _get_local_model(model_name: str) -> SentenceEncoder:
    """
    Loads and caches the local SentenceTransformer model.
    """
    global _local_model
    if _local_model is None:
        settings = get_settings()
        logger.info(
            "Initializing local embedding model",
            model_name=model_name,
            revision=settings.local_model_revision,
        )

        # Eliminate intermediate variable to prevent reportUnknownVariableType.
        # Cast the untyped instantiation directly to our Protocol.
        _local_model = cast(
            SentenceEncoder,
            SentenceTransformer(
                model_name_or_path=model_name,
                revision=settings.local_model_revision,
                trust_remote_code=False,
            ),
        )

    return _local_model


async def embed_texts(
    texts: list[str], model_name: str | None = None
) -> list[list[float]]:
    """
    Generates embeddings for a batch of texts.
    Automatically routes to OpenAI API or local model based on model_name prefix.
    """
    settings = get_settings()
    target_model = model_name or settings.embedding_model

    if target_model.startswith("text-embedding"):
        logger.debug("Generating embeddings via OpenAI", batch_size=len(texts))
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        response = await client.embeddings.create(input=texts, model=target_model)

        return [data.embedding for data in response.data]
    else:
        logger.debug("Generating embeddings locally", batch_size=len(texts))
        local_model = _get_local_model(target_model)

        def _encode_wrapper(texts_to_encode: list[str]) -> Any:
            return local_model.encode(texts_to_encode)

        embeddings: Any = await asyncio.to_thread(_encode_wrapper, texts)

        return [cast(list[float], emb.tolist()) for emb in embeddings]  # type: ignore[reportUnknownMemberType]


async def embed_query(query: str, model_name: str | None = None) -> list[float]:
    """
    Convenience method to embed a single query string.
    """
    embeddings = await embed_texts([query], model_name)
    return embeddings[0]
