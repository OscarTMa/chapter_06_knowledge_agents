import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from agent import KnowledgeRetrievalAgent

load_dotenv()

if __name__ == "__main__":
    # 1. Crear documento de muestra
    docs_folder = "./docs"
    os.makedirs(docs_folder, exist_ok=True)
    with open(os.path.join(docs_folder, "sample_policy.txt"), "w", encoding="utf-8") as f:
        f.write("Los agentes de IA del Capítulo 6 cubren recuperación de información, OCR inteligente y síntesis científica.")

    # 2. Iniciar agente y consultar
    agent = KnowledgeRetrievalAgent(docs_dir=docs_folder)
    agent.ingest_and_index()
    
    response = agent.query("¿De qué tratan los agentes del Capítulo 6?")
    print(f"\nRespuesta:\n{response.answer}\n")
    print(f"Fuentes utilizadas:\n{response.sources}")