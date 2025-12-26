from typing import List, Dict

class EvidenceConditioner:
    IMPORTANT_CUES = [
        "should", "should not", "recommended", "must",
        "avoid", "limit", "prefer", "increase", "reduce"
    ]

    MAX_CONTEXT_CHARS = 4000


    @classmethod
    def extract_guideline_sentences(cls, text: str) -> str:
        sentences = text.split(".")
        selected = [
            s.strip()
            for s in sentences
            if any(cue in s.lower() for cue in cls.IMPORTANT_CUES)
        ]
        return ". ".join(selected)


    @classmethod
    def condition_chunks(cls, chunks: List[Dict]) -> List[Dict]:
        conditioned = []

        for c in chunks:
            key_text = cls.extract_guideline_sentences(c["text"])

            if key_text:
                conditioned.append({
                    "content": key_text,
                    "source": c.get("id", "unknown")
                })

        return conditioned


    @classmethod
    def enforce_size_limit(cls, chunks: List[Dict]) -> List[Dict]:
        final = []
        total_chars = 0

        for c in chunks:
            size = len(c["content"])
            if total_chars + size > cls.MAX_CONTEXT_CHARS:
                break

            final.append(c)
            total_chars += size

        return final


    @classmethod
    def prepare_llm_context(cls, retrieved_chunks: List[Dict]) -> List[Dict]:
        conditioned = cls.condition_chunks(retrieved_chunks)
        final_context = cls.enforce_size_limit(conditioned)
        return final_context