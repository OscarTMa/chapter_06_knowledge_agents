import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

@dataclass
class RetrievalResponse:
    answer: str
    sources: List[str]

class KnowledgeRetrievalAgent:
    def __init__(self, docs_dir: str = "./docs", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 1. Embeddings locales ultrarrápidos (all-MiniLM-L6-v2)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. LLM Gemini
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def ingest_and_index(self):
        """Carga documentos locales y genera el índice FAISS."""
        documents = []
        for file_path in self.docs_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata={"source": str(file_path)}))
        
        if not documents:
            raise ValueError(f"No se encontraron archivos .txt en {self.docs_dir}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt_template = """Responde a la pregunta basándote únicamente en el siguiente contexto:
{context}

Pregunta: {question}
Respuesta:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_query: str) -> RetrievalResponse:
        """Ejecuta la consulta y retorna la respuesta con trazabilidad de fuentes."""
        if not self.rag_chain:
            raise ValueError("El agente no ha indexado documentos aún. Llame a ingest_and_index().")
        
        retrieved_docs = self.retriever.invoke(user_query)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in retrieved_docs]))
        answer = self.rag_chain.invoke(user_query)
        
        return RetrievalResponse(
            answer=answer,
            sources=sources
        )