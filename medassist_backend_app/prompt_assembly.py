from typing import List, Dict


class PromptAssembler:
    SYSTEM_PROMPT = """
You are MedAssist, a clinical guideline summarization assistant.

You must answer ONLY using the provided medical evidence.
If the evidence is insufficient or missing, say:
"Insufficient evidence in the provided guidelines."

Do NOT use prior medical knowledge.
Do NOT provide diagnosis.
Do NOT provide medication dosages.
If the question is outside clinical guidelines, say:
"Question outside clinical guidelines."
If the question indicates a medical emergency, say:
"Medical emergency detected. Please seek immediate medical attention."
"""

    ANSWER_CONSTRAINTS = """
Answer Guidelines:
- Use bullet points where appropriate
- Be concise and clinically neutral
- Do not invent recommendations
- Do not add external facts
- If recommendations vary, mention the variation
"""

    # ------------------------------------------------
    @classmethod
    def build_task_instruction(cls, user_query: str) -> str:
        return f"""
Task:
Answer the following clinical question based strictly on the evidence below.

Question:
{user_query}
"""

    # ------------------------------------------------
    @classmethod
    def build_evidence_block(cls, chunks: str) -> str:
        evidence_lines = []

        for i, chunk in enumerate(chunks, start=1):
            evidence_lines.append(
                f"[Evidence {i} | Source:{chunk}]"
            )

        return "\n\n".join(evidence_lines)

    # ------------------------------------------------
    @classmethod
    def assemble_prompt(
        cls,
        user_query: str,
        evidence_chunks: List[Dict],
        flag: str
    ) -> List[Dict]:

        return [
            {
                "role": "system",
                "content": cls.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    cls.build_task_instruction(user_query)
                    + "\n\nEvidence:\n"
                    + cls.build_evidence_block(evidence_chunks)
                    + "\n\n"
                    + cls.ANSWER_CONSTRAINTS
                    + f"\n\nFlag: {flag}"
                )
            }
        ]