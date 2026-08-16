import arxiv
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ScientificResearchAgent:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", n_clusters: int = 4):
        self.model = SentenceTransformer(embedding_model)
        self.n_clusters = n_clusters

    def scan_literature(self, query: str, max_results: int = 50) -> pd.DataFrame:
        """Fase 1: Escaneo y recolección de metadatos de literatura en arXiv."""
        results = []
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for r in search.results():
            if r.summary:
                results.append({
                    "title": r.title.strip(),
                    "summary": r.summary.strip().replace("\n", " "),
                    "authors": ", ".join(a.name for a in r.authors),
                    "published": r.published.strftime("%Y-%m-%d"),
                    "url": r.entry_id
                })
        return pd.DataFrame(results)

    def cluster_themes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fase 2: Vectorización de resúmenes y clustering temático."""
        if df.empty:
            return df
        
        embeddings = self.model.encode(df["summary"].tolist(), show_progress_bar=False)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(embeddings)
        return df

    def synthesize_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fase 3: Generación de tabla de evidencia y síntesis estructurada."""
        if df.empty:
            return {"status": "No data found"}

        synthesis_report = {}
        for c in range(self.n_clusters):
            cluster_papers = df[df["cluster"] == c]
            if cluster_papers.empty:
                continue
            
            top_papers = cluster_papers.sort_values("published", ascending=False).head(3)
            synthesis_report[f"Cluster_{c}"] = {
                "paper_count": len(cluster_papers),
                "representative_papers": top_papers[["title", "authors", "published", "url"]].to_dict(orient="records")
            }
            
        return synthesis_report