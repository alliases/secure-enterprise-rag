# File: app/llm/provider.py
# Purpose: LLM provider abstraction with exponential backoff retry logic.

from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.logging_config.setup import get_logger

logger = get_logger(__name__)


# Retry strategy: wait 2^x * 1 seconds between each retry, up to 10 seconds max, max 3 attempts.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError)),
    reraise=True,
)
async def get_llm_response(
    system_prompt: str,
    user_message: str,
    context_chunks: list[str],
) -> str:
    """
    Sends the masked query and retrieved context to the LLM.
    Protected by resilience logic to handle temporary cloud provider outages.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Compile the strict boundary prompt
    context_str = "\n\n---\n\n".join(context_chunks)
    full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context_str}"

    logger.info("Executing LLM generation", model=settings.llm_model)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Low temperature for analytical consistency (reduced hallucination risk)
        max_tokens=1024,
    )

    result = response.choices[0].message.content
    return result or "Error: Empty response from LLM"
