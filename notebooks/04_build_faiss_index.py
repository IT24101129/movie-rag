import numpy as np
import faiss

# ── Load embeddings ───────────────────────────────────────────────────────────
embeddings = np.load("data/movie_embeddings.npy")
print(f"Loaded embeddings: {embeddings.shape}")

dimension = embeddings.shape[1]  # 384
n_vectors = embeddings.shape[0]  # 4799

# ── Build FAISS index ─────────────────────────────────────────────────────────
index = faiss.IndexFlatL2(dimension)
print(f"Created empty index. Dimension: {dimension}")

index.add(embeddings)
print(f"Added {index.ntotal} vectors to the index")

# ── Sanity check ──────────────────────────────────────────────────────────────
# Search using Avatar's own vector — should match itself first with distance ~0
query_vector = embeddings[0:1]
distances, indices = index.search(query_vector, k=5)

print("\n── Sanity check: Avatar's 5 nearest neighbors ──")
print(f"Indices : {indices[0]}")
print(f"Distances: {distances[0]}")
print("(Index 0 should be first with distance ~0.0)")

# ── Save ──────────────────────────────────────────────────────────────────────
faiss.write_index(index, "data/movie_index.faiss")
print("\n✅ Saved → data/movie_index.faiss")