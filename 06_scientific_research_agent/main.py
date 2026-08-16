from agent import ScientificResearchAgent

agent = ScientificResearchAgent(n_clusters=3)
print("Escaneando arXiv...")
df = agent.scan_literature("retrieval augmented generation llm", max_results=10)
df_clustered = agent.cluster_themes(df)
synthesis = agent.synthesize_insights(df_clustered)

print("\n--- Síntesis Generada ---")
for cluster, data in synthesis.items():
    print(f"\n{cluster}: {data['paper_count']} papers")
    for paper in data['representative_papers']:
        print(f" - {paper['title']} ({paper['published']})")