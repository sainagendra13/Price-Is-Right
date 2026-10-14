import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@lru_cache
def get_client():
    if os.getenv("LLM_PROVIDER", "openai").lower() == "ollama":
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Add OPENAI_API_KEY to .env, "
            "or set LLM_PROVIDER=ollama"
        )

    return OpenAI()


def ask(system, user, as_json=False):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    model = (
        os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        if provider == "ollama"
        else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    reply = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        **(
            {"response_format": {"type": "json_object"}}
            if as_json else {}
        )
    )

    return (reply.choices[0].message.content or "").strip()