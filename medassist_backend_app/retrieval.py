import os
from dotenv import load_dotenv
from typing import List, Dict

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


class HybridRetriever:
    def __init__(self):
        load_dotenv()

        self.AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
        self.AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

        self.AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        self.AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
        self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

       
        self.search_client = SearchClient(
            endpoint=self.AZURE_SEARCH_ENDPOINT,
            index_name=self.AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(self.AZURE_SEARCH_KEY),
        )

        self.openai_client = AzureOpenAI(
            azure_endpoint=self.AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=self.AZURE_OPENAI_EMBEDDING_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
        )

    
    def embed_query(self, query: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=query,
        )
        return response.data[0].embedding

    
    def vector_search(self, query_embedding: List[float], k: int = 10) -> Dict[str, Dict]:
        results = self.search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "contentVector",
                "k": k
            }],
            select=["id", "content", "contentVector", "source", "year"]
        )

        vector_hits = {}
        for r in results:
            vector_hits[r["id"]] = {
                "text": r["content"],
                "embedding": r["contentVector"],
                "vector_score": r["@search.score"],
                "source": r.get("source"),
                "year": r.get("year")
            }

        return vector_hits

    
    def keyword_search(self, query: str, k: int = 10) -> Dict[str, Dict]:
        results = self.search_client.search(
            search_text=query,
            top=k,
            select=["id", "content", "source", "year"]
        )

        keyword_hits = {}
        for r in results:
            keyword_hits[r["id"]] = {
                "text": r["content"],
                "bm25_score": r["@search.score"],
                "source": r.get("source"),
                "year": r.get("year")
            }

        return keyword_hits

    
    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:

        query_embedding = self.embed_query(query)

        vector_results = self.vector_search(query_embedding)
        keyword_results = self.keyword_search(query)

        all_doc_ids = set(vector_results) | set(keyword_results)

        max_vector = max(
            (v["vector_score"] for v in vector_results.values()),
            default=1.0
        )
        max_bm25 = max(
            (v["bm25_score"] for v in keyword_results.values()),
            default=1.0
        )

        merged = []

        for doc_id in all_doc_ids:
            v = vector_results.get(doc_id, {})
            k = keyword_results.get(doc_id, {})

            norm_vector = (v.get("vector_score", 0) / max_vector) if max_vector else 0
            norm_bm25 = (k.get("bm25_score", 0) / max_bm25) if max_bm25 else 0

            hybrid_score = 0.6 * norm_vector + 0.4 * norm_bm25

            merged.append({
                "id": doc_id,
                "text": v.get("text") or k.get("text"),
                "embedding": v.get("embedding"),
                "source": v.get("source") or k.get("source"),
                "year": v.get("year") or k.get("year"),
                "score": hybrid_score
            })

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]