import os
import base64
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


# ============================================================
# Load environment variables
# ============================================================
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

PDF_DIR = "./pdfs"


# ============================================================
# Clients
# ============================================================
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY),
)

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_key=AZURE_OPENAI_EMBEDDING_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


# ============================================================
# Utility functions
# ============================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding


def safe_id(raw_text: str) -> str:
    """
    Azure AI Search–safe document ID:
    Base64 URL-safe encoding (no dots, slashes, etc.)
    """
    return base64.urlsafe_b64encode(
        raw_text.encode("utf-8")
    ).decode("utf-8")


# ============================================================
# Ingestion logic
# ============================================================
def ingest_pdf(pdf_path: str):
    print(f"\n📄 Processing PDF: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("⚠️ No text extracted, skipping.")
        return

    chunks = chunk_text(text)
    print(f"🔹 Total chunks created: {len(chunks)}")

    documents = []

    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk)

            raw_id = f"{pdf_path}-{i}"
            document = {
                "id": safe_id(raw_id),
                "content": chunk,
                "contentVector": embedding,
                "source": "WHO,CDC,NIH",  # Example source
                "year": 2024  # Example year
            }

            documents.append(document)

        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")

    if documents:
        result = search_client.upload_documents(documents)
        print(f"✅ Uploaded {len(documents)} chunks to Azure AI Search")
    else:
        print("⚠️ No documents to upload.")


# ============================================================
# Main runner
# ============================================================
def main():
    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for pdf_file in pdf_files:
        ingest_pdf(os.path.join(PDF_DIR, pdf_file))

    print("\n🎉 Ingestion completed successfully.")


if __name__ == "__main__":
    main()
