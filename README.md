# MedAssist – Medical RAG System (Full-Stack AI Project)

## Overview

**MedAssist** is a **production-oriented Medical Retrieval-Augmented Generation (RAG) system** built using **Django, React, and Azure AI services**.
The system is designed to deliver **evidence-grounded medical responses**, enforce **safety guardrails**, and prevent **LLM hallucinations** in healthcare use cases.

This project demonstrates **end-to-end AI system engineering**, including backend APIs, frontend UI, cloud AI integration, and responsible AI design.

---

## Core Skills Demonstrated (ATS-Optimized)

* Retrieval-Augmented Generation (RAG)
* Full-Stack Development (Django + React)
* Azure Cloud Services
* Azure OpenAI (GPT-4o)
* Vector Search & Text Embeddings
* Azure AI Search
* REST API Design
* Secure Prompt Engineering
* Responsible / Safe AI Systems
* Healthcare AI Architecture
* Production-ready System Design

---

## Key Functional Capabilities

* Evidence-based medical question answering
* Vector-based semantic retrieval
* Controlled prompt assembly
* Emergency intent detection
* Out-of-scope query handling
* Hallucination prevention
* Frontend–backend API integration

---

## Safety & Responsible AI Design

The system enforces **explicit safety logic**, critical for healthcare AI applications.

* **Medical Emergency Detection**
  If emergency intent is detected, the system returns:

  > *Medical emergency detected. Please seek immediate professional help.*

* **Insufficient Data Handling**
  If no relevant documents are retrieved from the vector index:

  > *Insufficient data available to provide a reliable answer.*

The model **never generates answers outside indexed medical evidence**.

---

## Technical Architecture (Recruiter-Friendly)

### Backend

* Django
* Django REST Framework
* Python
* REST APIs
* Session & request handling

### Frontend

* React
* JavaScript
* Chat-based UI
* Typing indicators
* Input control & validation

### Cloud & AI

* Azure OpenAI – **GPT-4o**
* Azure OpenAI – **Text Embedding Model**
* Azure AI Search – Vector indexing & retrieval
* Azure-based RAG architecture

### AI Pipeline

1. Document ingestion & chunking
2. Text embedding generation
3. Vector indexing (Azure AI Search)
4. Semantic retrieval
5. Evidence filtering
6. Prompt assembly
7. Controlled response generation

---

## Backend – Local Setup (Django API)

### Prerequisites

* Python 3.10+
* Azure OpenAI Resource
* Azure AI Search Resource

### Run Backend Locally

```bash
git clone https://github.com/thilak-sekhar/medassist_backend.git
cd medassist_backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Environment Variables

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-model
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX=
```

Backend runs at:

```
http://127.0.0.1:8000/
```

---

## Frontend – Local Setup (React UI)

### Prerequisites

* Node.js 18+
* npm

### Run Frontend Locally

```bash
git clone https://github.com/thilak-sekhar/medassist_frontend.git
cd medassist-frontend
npm install
npm start
```

Update backend URL in `App.js`:

```js
const apiUrl = "http://127.0.0.1:8000/chat/";
```

Frontend runs at:

```
http://localhost:3000/
```

---

## API Example

```http
POST /chat/
Content-Type: application/json

{
  "query": "dietary recommendations for diabetic patients"
}
```

---

## Why This Project Matters

* Demonstrates **applied AI engineering**, not just model usage
* Shows **cloud AI integration with Azure**
* Implements **RAG best practices**
* Enforces **responsible AI constraints**
* Built with **real deployment considerations**
* Relevant to **AI Engineer, ML Engineer, Backend Engineer, Full-Stack Engineer** roles

---

## Intended Roles 

* AI Engineer
* Machine Learning Engineer
* Applied ML Engineer
* Backend Engineer
* Full-Stack Developer
* Cloud AI Engineer
* Healthcare AI Engineer

---

## Disclaimer

This project is for **educational and research purposes only** and is **not a replacement for professional medical advice**.

---

### Optional Next Improvements (If You Want)

* Add **architecture diagram**
* Add **system flow diagram**
* Add **performance metrics**
* Add **unit / integration tests**
* Add **deployment guide (Render / Azure App Service)**
