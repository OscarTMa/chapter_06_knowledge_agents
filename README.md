# Chapter 06: Information Retrieval & Knowledge Agents

Este módulo implementa la tríada de agentes de conocimiento descritos en el libro *30 Agents Every AI Engineer Must Build* (Capítulo 6). Estos agentes transforman modelos de lenguaje estáticos en sistemas fundamentados en evidencia empírica, estructuración de documentos y síntesis científica a gran escala.

---

## Arquitectura y Flujo de los Agentes

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingesta y Percepción"]
        A1[Archivos de Texto / Docs] --> B1[Knowledge Retrieval Agent]
        A2[Imágenes / PDFs Escaneados] --> B2[Document Intelligence Agent]
        A3[APIs Académicas / arXiv] --> B3[Scientific Research Agent]
    end

    subgraph Processing["2. Procesamiento y Razonamiento"]
        B1 --> C1[Chunking + Bi-Encoder Embeddings]
        C1 --> D1[(FAISS Vector DB)]
        
        B2 --> C2[Tesseract OCR + Bounding Boxes]
        C2 --> D2[Fuzzy Schema Matching + Confidence Filter]
        
        B3 --> C3[Sentence Transformers Embeddings]
        C3 --> D3[K-Means Thematic Clustering]
    end

    subgraph Grounding["3. Generación y Síntesis Grounded"]
        D1 --> E1[LLM Gemini Ingestion & Decodificación Greedy]
        D2 --> E2[Extracción Estructurada JSON con Provenance]
        D3 --> E3[Tablas de Evidencia y Mapeo de Consenso]
    end

    E1 --> Output1[Respuesta con Citas de Fuentes]
    E2 --> Output2[Datos Listos para ERP / CRM / BD]
    E3 --> Output3[Reporte de Síntesis y Brechas de Conocimiento]
```
---
## Comparativa de los Agentes

| # | Agente | Rol Principal | Nivel de Capacidad | Tecnologías Clave |
| :--- | :--- | :--- | :--- |
| **04** | |Knowledge Retrieval AgentConectar LLMs con fuentes vivas mitigando alucinaciones (RAG denso)Nivel 2–3 (Tool-Using / Planning)LangChain LCEL, FAISS, Hugging Face, Google Gemini
| **05** |Document Intelligence AgentExtraer datos tabulares y campos clave de documentos no estructuradosNivel 2–3 (Tool-Using / Planning)Tesseract OCR, Pillow, RapidFuzz
| **06** | Scientific Research AgentEscaneo masivo, agrupamiento semántico y síntesis de literaturaNivel 4 (Learning & Discovery)arXiv API, SentenceTransformers, Scikit-Learn, PandasEstructura del RepositorioPlaintext

chapter_06_knowledge_agents/
├── .env.example
├── .gitignore
├── requirements.txt

├── README.md
├── 04_knowledge_retrieval_agent/
│   ├── docs/
│   │   └── sample_policy.txt
│   ├── agent.py
│   └── main.py
├── 05_document_intelligence_agent/
│   ├── agent.py
│   └── main.py
└── 06_scientific_research_agent/
    ├── agent.py
    └── main.py

Requisitos del Sistema y Configuración

1. Dependencias del Sistema Operativo (Linux / Ubuntu)Para el motor de OCR del Agente 05:Bashsudo apt update
sudo apt install -y tesseract-ocr poppler-utils

2. Configuración del Entorno Virtual PythonBashpython3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

3. Variables de EntornoCopia el archivo .env.example a .env y configura tu clave de API:Bashcp .env.example .env
Contenido del .env:Extrait de codeGOOGLE_API_KEY="tu_google_gemini_api_key"
Ejecución de Pruebas

Agente 04 (Knowledge Retrieval):Bashpython

04_knowledge_retrieval_agent/main.py

Agente 05 (Document Intelligence):Bashpython 

05_document_intelligence_agent/main.py

Agente 06 (Scientific Research):Bashpython 

06_scientific_research_agent/main.py
