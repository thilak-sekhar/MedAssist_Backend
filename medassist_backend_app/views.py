from rest_framework.views import APIView
from rest_framework.response import Response
from .query_classifier import QueryClassifier
from .retrieval import HybridRetriever
from .evidence_conditioning import EvidenceConditioner
from .rerank_and_context import MedicalReranker
from .prompt_assembly import PromptAssembler
from .model_generation import FinalAnswerGenerator

class ChatView(APIView):
    def post(self, request):
        query = request.data.get("query")
        
        classifier = QueryClassifier()
        flag = classifier.classify_query(query)

        retrieval = HybridRetriever()
        documents = retrieval.hybrid_retrieval(query, top_k=10)

        evidence_docs = EvidenceConditioner.prepare_llm_context(documents)

        rerank = MedicalReranker()
        reranked_chunks = rerank.medical_rerank(query, evidence_docs, top_k=3)
        final_context = MedicalReranker.build_context(reranked_chunks)

        prompt = PromptAssembler.assemble_prompt(query, evidence_docs, flag)

        final_answer = FinalAnswerGenerator.generate_final_answer(prompt)

        return Response({"answer":final_answer})
