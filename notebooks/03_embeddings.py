import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# ── Load enriched data ────────────────────────────────────────────────────────
df = pd.read_csv("data/enriched_movies.csv")
print(f"Loaded {len(df)} movies")

# Convert string-lists back to actual lists
for col in ["genres", "keywords", "cast"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# ── Build a single text blob per movie ────────────────────────────────────────
def build_text_blob(row):
    parts = [
        row["title"],
        row["overview"],
        "Genres: " + ", ".join(row["genres"]),
        "Mood: " + row["mood"],
        "Pacing: " + row["pacing"],
        "Attention required: " + row["attention_required"],
        "Emotional intensity: " + row["emotional_intensity"],
        "Family friendly: " + row["family_friendly"],
        "Good to watch while eating: " + row["watch_with_food"],
    ]
    return ". ".join(str(p) for p in parts if p)

df["text_blob"] = df.apply(build_text_blob, axis=1)

print("\n── Sample text blob ──")
print(df["text_blob"].iloc[0])

# ── Load the embedding model ──────────────────────────────────────────────────
print("\nLoading sentence-transformers model (this may take a moment the first time)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

# ── Generate embeddings ────────────────────────────────────────────────────────
print("\nGenerating embeddings for all movies...")
start = time.time()

texts = df["text_blob"].tolist()
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f} seconds")
print(f"Embeddings shape: {embeddings.shape}")  # should be (4799, 384)

# ── Save embeddings + corresponding dataframe ─────────────────────────────────
np.save("data/movie_embeddings.npy", embeddings)
df.to_csv("data/movies_with_blobs.csv", index=False)

print("\n✅ Saved → data/movie_embeddings.npy")
print("✅ Saved → data/movies_with_blobs.csv")