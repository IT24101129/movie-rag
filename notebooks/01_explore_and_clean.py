import pandas as pd
import numpy as np
import ast
import os

# ── Load ──────────────────────────────────────────────────────────────────────
movies  = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print("Movies shape :", movies.shape)
print("Credits shape:", credits.shape)
print("\n── Movies columns ──")
print(movies.columns.tolist())
print("\n── Credits columns ──")
print(credits.columns.tolist())

# ── Merge ─────────────────────────────────────────────────────────────────────
# credits uses 'movie_id'; rename to match movies' 'id'
credits = credits.rename(columns={"movie_id": "id"})
df = movies.merge(credits, on="id", how="left")
print(f"\nAfter merge: {df.shape}")

# ── Drop columns we won't use ─────────────────────────────────────────────────
drop_cols = [
    "homepage", "spoken_languages", "status",
    "original_title", "title_y"          # title_y is duplicate from credits
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# rename title_x → title if it exists
if "title_x" in df.columns:
    df = df.rename(columns={"title_x": "title"})

# ── Parse JSON-string columns ─────────────────────────────────────────────────
def parse_names(json_str, key="name", limit=None):
    """Extract a list of 'name' values from a JSON-encoded string."""
    try:
        items = ast.literal_eval(json_str)
        names = [item[key] for item in items if key in item]
        return names[:limit] if limit else names
    except Exception:
        return []

df["genres"]            = df["genres"].apply(parse_names)
df["keywords"]          = df["keywords"].apply(parse_names)
df["production_companies"] = df["production_companies"].apply(parse_names)

# Cast: top 5 actor names
df["cast"] = df["cast"].apply(lambda x: parse_names(x, limit=5))

# Crew: director only
def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "")
    except Exception:
        pass
    return ""

df["director"] = df["crew"].apply(get_director)
df = df.drop(columns=["crew"])

# ── Null audit ────────────────────────────────────────────────────────────────
print("\n── Missing values ──")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── Fill / drop nulls ─────────────────────────────────────────────────────────
df["overview"]         = df["overview"].fillna("")
df["tagline"]          = df["tagline"].fillna("")
df["runtime"]          = df["runtime"].fillna(df["runtime"].median())
df["release_date"]     = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"]     = df["release_date"].dt.year.fillna(0).astype(int)

# Drop rows with no title or overview (unusable for RAG)
before = len(df)
df = df.dropna(subset=["title", "overview"])
df = df[df["overview"].str.strip() != ""]
print(f"\nDropped {before - len(df)} rows with missing title/overview")

# ── Type fixes ────────────────────────────────────────────────────────────────
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0).astype(int)
df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0.0)
df["budget"]       = pd.to_numeric(df["budget"],       errors="coerce").fillna(0)
df["revenue"]      = pd.to_numeric(df["revenue"],      errors="coerce").fillna(0)

# ── Keep only useful columns ──────────────────────────────────────────────────
keep_cols = [
    "id", "title", "overview", "tagline",
    "genres", "keywords", "cast", "director",
    "production_companies", "release_year",
    "runtime", "vote_average", "vote_count", "popularity",
    "budget", "revenue", "original_language"
]
df = df[[c for c in keep_cols if c in df.columns]]

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n── Final dataset ──")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample row:")
print(df.iloc[0].to_dict())

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_movies.csv", index=False)
print("\n✅ Saved → data/cleaned_movies.csv")