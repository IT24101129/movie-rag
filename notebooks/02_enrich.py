import pandas as pd
import ast

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_movies.csv")

# Convert string representations of lists back to actual lists
for col in ["genres", "keywords", "cast"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# ── Helper ────────────────────────────────────────────────────────────────────
def has_any(tags, *terms):
    """Return True if any term appears in the tags list (case-insensitive)."""
    tags_lower = [t.lower() for t in tags]
    return any(term.lower() in tags_lower for term in terms)

# ══════════════════════════════════════════════════════════════════════════════
# 1. MOOD
# ══════════════════════════════════════════════════════════════════════════════
def assign_mood(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k

    # Action/war checked first to avoid misclassification
    if has_any(combined, "Horror", "fear", "terrifying", "haunting", "supernatural"):
        return "dark"
    if has_any(combined, "Thriller", "suspense", "tension", "conspiracy"):
        return "tense"
    if has_any(combined, "Action", "war", "battle", "fight", "explosion", "crime"):
        return "exciting"
    if has_any(combined, "Comedy", "funny", "humor", "satire", "parody"):
        return "happy"
    if has_any(combined, "Romance", "love", "romantic", "wedding"):
        return "romantic"
    if has_any(combined, "Animation", "Family", "fairy tale", "talking animal"):
        return "whimsical"
    if has_any(combined, "Documentary", "biography", "history", "true story"):
        return "thoughtful"
    if has_any(combined, "Drama", "grief", "loss", "emotional", "touching"):
        return "emotional"
    return "neutral"

# ══════════════════════════════════════════════════════════════════════════════
# 2. PACING
# ══════════════════════════════════════════════════════════════════════════════
def assign_pacing(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k
    runtime = row["runtime"]

    if has_any(combined, "Action", "Thriller", "chase", "race", "fight", "explosion"):
        return "fast"
    if has_any(combined, "Drama", "Documentary", "slow burn", "character study", "biographical"):
        return "slow"
    if runtime < 90:
        return "fast"
    if runtime > 140:
        return "slow"
    return "moderate"

# ══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION REQUIRED
# ══════════════════════════════════════════════════════════════════════════════
def assign_attention(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k

    if has_any(combined, "Mystery", "plot twist", "nonlinear", "mindbending",
               "psychological", "complex narrative", "Science Fiction"):
        return "high"
    if has_any(combined, "Comedy", "Animation", "Romance", "funny", "lighthearted"):
        return "low"
    return "medium"

# ══════════════════════════════════════════════════════════════════════════════
# 4. EMOTIONAL INTENSITY
# ══════════════════════════════════════════════════════════════════════════════
def assign_emotional_intensity(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k

    if has_any(combined, "Horror", "War", "tragedy", "grief", "death",
               "abuse", "loss", "trauma", "heartbreak"):
        return "high"
    if has_any(combined, "Comedy", "Animation", "Adventure", "funny", "lighthearted"):
        return "low"
    return "medium"

# ══════════════════════════════════════════════════════════════════════════════
# 5. FAMILY FRIENDLY
# ══════════════════════════════════════════════════════════════════════════════
def assign_family_friendly(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k

    if has_any(combined, "Animation", "Family", "children", "kids", "fairy tale", "talking animal"):
        return "yes"
    if has_any(combined, "Horror", "adult", "erotic", "violence", "gore",
               "drug", "nudity", "sexual"):
        return "no"
    return "yes" if row["vote_average"] >= 6.0 else "no"

# ══════════════════════════════════════════════════════════════════════════════
# 6. WATCH WITH FOOD
# ══════════════════════════════════════════════════════════════════════════════
def assign_watch_with_food(row):
    g = row["genres"]
    k = row["keywords"]
    combined = g + k

    # Easy to follow while eating = low attention, light mood
    if has_any(combined, "Comedy", "Animation", "Romance", "Family",
               "funny", "lighthearted", "feel-good"):
        return "yes"
    # Hard to follow while eating = intense or complex
    if has_any(combined, "Horror", "Thriller", "Mystery", "War",
               "psychological", "plot twist", "gore"):
        return "no"
    return "yes"

# ── Apply all functions ───────────────────────────────────────────────────────
print("Enriching dataset...")

df["mood"]               = df.apply(assign_mood, axis=1)
df["pacing"]             = df.apply(assign_pacing, axis=1)
df["attention_required"] = df.apply(assign_attention, axis=1)
df["emotional_intensity"]= df.apply(assign_emotional_intensity, axis=1)
df["family_friendly"]    = df.apply(assign_family_friendly, axis=1)
df["watch_with_food"]    = df.apply(assign_watch_with_food, axis=1)

# ── Spot check ────────────────────────────────────────────────────────────────
print("\n── Sample enriched rows ──")
sample_cols = ["title", "genres", "mood", "pacing", "attention_required",
               "emotional_intensity", "family_friendly", "watch_with_food"]
print(df[sample_cols].head(10).to_string(index=False))

# ── Distribution check ────────────────────────────────────────────────────────
print("\n── Mood distribution ──")
print(df["mood"].value_counts())

print("\n── Pacing distribution ──")
print(df["pacing"].value_counts())

print("\n── Attention Required distribution ──")
print(df["attention_required"].value_counts())

print("\n── Emotional Intensity distribution ──")
print(df["emotional_intensity"].value_counts())

print("\n── Family Friendly distribution ──")
print(df["family_friendly"].value_counts())

print("\n── Watch With Food distribution ──")
print(df["watch_with_food"].value_counts())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("data/enriched_movies.csv", index=False)
print("\n✅ Saved → data/enriched_movies.csv")