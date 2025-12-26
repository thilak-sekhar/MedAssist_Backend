import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import AzureOpenAI


class MedicalReranker:
    def __init__(self):
        load_dotenv()

        self.AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        self.AZURE_OPENAI_CHAT_KEY = os.getenv("AZURE_OPENAI_CHAT_KEY")
        self.AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


        self.chat_client = AzureOpenAI(
            azure_endpoint=self.AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=self.AZURE_OPENAI_CHAT_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
        )


    def medical_rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:

        reranked = []

        for chunk in chunks:
            prompt = f"""
You are a medical expert.

Rate how clinically relevant the following text is for answering the question.

Question:
{query}

Text:
{chunk["content"]}

Respond with ONLY a number from 0 to 10.
"""

            response = self.chat_client.chat.completions.create(
                model=self.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict medical relevance evaluator. "
                            "Respond with only a number."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            score_text = response.choices[0].message.content.strip()

            try:
                score = float(score_text)
            except ValueError:
                score = 0.0

            reranked.append({
                "content": chunk["content"],
                "score": score
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]


    @staticmethod
    def build_context(chunks: List[Dict], max_chars: int = 2000) -> str:
        context = ""
        for c in chunks:
            if len(context) + len(c["content"]) > max_chars:
                break
            context += c["content"] + "\n\n"
        return context.strip()
