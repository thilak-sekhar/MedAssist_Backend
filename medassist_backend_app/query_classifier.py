import os
from dotenv import load_dotenv
from openai import AzureOpenAI


class QueryClassifier:
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

        self.EMERGENCY_TERMS = [
            "chest pain", "shortness of breath", "difficulty breathing",
            "loss of consciousness", "seizure", "severe bleeding",
            "stroke", "heart attack", "sudden weakness"
        ]

        self.DIET_TERMS = [
            "diet", "food", "eat", "nutrition", "meal", "dietary"
        ]

        self.LIFESTYLE_TERMS = [
            "exercise", "physical activity", "sleep", "stress",
            "lifestyle", "habits"
        ]

        self.DISEASE_TERMS = [
            "diabetes", "hypertension", "asthma",
            "heart disease", "thyroid", "cancer"
        ]

        self.SYMPTOM_TERMS = [
            "symptoms", "pain", "fever", "cough", "headache",
            "nausea", "vomiting", "dizziness"
        ]

        self.GENERAL_EDUCATION_TERMS = [
            "what is", "information about", "tell me about",
            "explain", "define", "treatments"
        ]

    def classify_query_rule_based(self, query: str) -> str | None:
        q = query.lower()

        if any(term in q for term in self.EMERGENCY_TERMS):
            return "emergency_flag"

        if any(term in q for term in self.DIET_TERMS):
            return "dietary_guidance"

        if any(term in q for term in self.LIFESTYLE_TERMS):
            return "lifestyle_advice"

        if any(term in q for term in self.DISEASE_TERMS):
            return "disease_management"

        if any(term in q for term in self.SYMPTOM_TERMS):
            return "symptom_check"

        if any(term in q for term in self.GENERAL_EDUCATION_TERMS):
            return "general_education"

        return None

    def classify_query_llm(self, query: str) -> str | None:
        INTENT_SYSTEM_PROMPT = """
You are a medical query classifier.

You MUST return exactly ONE label from this list:
dietary_guidance
lifestyle_advice
disease_management
symptom_check
emergency_flag
general_education

Rules:
- Output ONLY the label
- No punctuation
- No explanation
- No extra text
- If the query indicates a potential medical emergency,
  classify it as "emergency_flag"
"""

        response = self.chat_client.chat.completions.create(
            model=self.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0
        )

        content = response.choices[0].message.content
        return content.strip().lower() if content else None

    def classify_query(self, query: str) -> str:
        rule_intent = self.classify_query_rule_based(query)

        if rule_intent:
            return rule_intent

        llm_intent = self.classify_query_llm(query)
        return llm_intent if llm_intent else "unknown"
