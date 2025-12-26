import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import List, Dict

load_dotenv()


class FinalAnswerGenerator:

    AZURE_OPENAI_API_VERSION = "2024-06-01"
    AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
    AZURE_OPENAI_CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_KEY")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


    UNSAFE_TERMS = [
        "diagnose", "dosage", "mg",
        "operate", "inject", "treatment plan"
    ]


    chat_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
        api_key=AZURE_OPENAI_CHAT_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


    @classmethod
    def contains_unsafe_terms(cls, text: str) -> bool:
        return any(term in text.lower() for term in cls.UNSAFE_TERMS)


    @classmethod
    def generate_final_answer(
        cls,
        prompt_messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 450
    ) -> str:

        response = cls.chat_client.chat.completions.create(
            model=cls.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content.strip()

        # Optional post-generation safety gate
        if cls.contains_unsafe_terms(answer):
            return (
                "The response contained restricted medical content and "
                "cannot be displayed."
            )

        return answer
