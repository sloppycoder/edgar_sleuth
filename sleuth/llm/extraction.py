import json
import logging
import os
from typing import Any, Optional

from vertexai.generative_models import GenerativeModel

from .util import init_vertaxai, openai_client

logger = logging.getLogger(__name__)

_DUMMY_RESPONSE = """```json
{
    "notes":"extraction skippped",
    "trustees":[],
    "compensation_info_present":false
 }
```
"""


def ask_model(model: str, prompt: str) -> Optional[str]:
    # provides a mechanism to skip asking the model
    # in case it's not need for testing and saves cost
    if os.environ.get("SKIP_ASK_MODEL", "0") == "1":
        return _DUMMY_RESPONSE

    if model.startswith("gemini"):
        return _chat_with_gemini(model, prompt)
    elif model.startswith("gpt"):
        return _chat_with_gpt(model, prompt)
    else:
        raise ValueError(f"Unknown model: {model}")


def extract_json_from_response(response: str) -> dict[str, Any]:
    # the response should be a JSON
    # sometimes Gemini wraps it in a markdown block ```json ...```
    # so we unrap the markdown block and get to the json
    if len(response) > 10:
        start_markdown_index = response.find("```json")
        end_markdown__index = response.rfind("```")
        if start_markdown_index >= 0 and end_markdown__index >= 0:
            try:
                return json.loads(
                    response[start_markdown_index + 7 : end_markdown__index]
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response")
    return {}


def _chat_with_gpt(model_name: str, prompt: str) -> Optional[str]:
    client = openai_client()

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.warning(f"Error calling OpenAI API: {str(e)}")
        return None


def _chat_with_gemini(model_name: str, prompt: str) -> Optional[str]:
    try:
        init_vertaxai()
        model = GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0,
                "top_p": 0.95,
            },
        )
        return response.text
    except Exception as e:
        logging.warning(f"Error calling Google Gemini API: {str(e)}")
        return None
