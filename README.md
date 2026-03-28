# Mobile App Competitor Sub-niche Discovery

Automated pipeline for grouping ~4 200 iOS subscription apps into **competitor sub-niches** — clusters of apps that directly compete for the same user need.

> **Final result:** [`app_sub_niches.csv`](app_sub_niches.csv) — each app paired with its sub-niche name and description.

---

## Project Structure

```
.
├── subscription_apps.json        # Raw dataset — one iOS app per row
│
├── kmeans.ipynb                  # Approach 1: KMeans clustering
├── NearestNeighbors.ipynb        # Approach 2: KNN + graph connected components  ← selected
├── topic_modeling.ipynb          # Approach 3: BERTopic (UMAP + HDBSCAN + c-TF-IDF)
├── llm_validation.ipynb          # Stage 2: LLM sub-niche refinement (runed on top of NN clusters)
│
├── embeddings_features.npy       # Pre-computed sentence embeddings (BAAI/bge-base-en)
├── clustered_apps_NN.csv         # Output of NearestNeighbors.ipynb — apps with cluster labels
├── app_sub_niches.csv            # ✅ FINAL OUTPUT — apps with sub-niche name and description
│
├── requirements.txt
└── .env                          # OPENAI_API_KEY (not committed)
```

---

## My Thinking

### Why these three approaches?

When I started this task I genuinely wasn't sure which clustering strategy would work best for app data, so I decided to try all three that I had experience with and compare the results rather than commit to one upfront.

I started with **KMeans** because it's the most familiar and fastest to run — it gave me a quick baseline to look at and reason about. But when I actually inspected the clusters, I kept finding apps that didn't really belong together — things that were close related but clearly not competitors. On top of that, choosing `k` felt arbitrary: I settled on 700 based on "~6 apps per cluster on average".

Then I tried **BERTopic** because it sounded like the most elegant solution — it clusters *and* labels the clusters automatically. I liked the idea of getting human-readable topic names without any extra work. But when I ran it, about 16% of apps ended up as outliers (topic `−1`). That's nearly one in six apps just... dropped from the analysis. For a task where the goal is to cover the whole market, that felt like too big a blind spot to accept. I tried tuning `min_cluster_size` and `min_samples` but the outlier count didn't improve enough to change my mind.

The **KNN graph approach** felt right as soon as I looked at the output. The idea of connecting apps that are genuinely close to each other and then finding natural groups through connected components just made intuitive sense for this problem. The threshold parameter is something I can actually reason about: "these apps share at least 80% directional similarity" is a much more grounded statement than "they're in the same KMeans bucket." The clusters were tight and, when I spot-checked them, apps in the same component really did feel like they belonged together.

### Why add an LLM on top?

After I got the KNN clusters I noticed something: even a tight, high-quality cluster can mix apps that *feel* similar but aren't actually competing. For example, an "AI image generation" cluster might contain a portrait photo app, a logo maker, and an AI wallpaper app — all use the same underlying technology but nobody selling wallpapers is competing with a logo designer. An embedding model can't make that distinction because it was never trained to think about market competition. It just knows that these texts are similar.

The LLM step felt like the natural solution. Instead of trying to engineer that judgment into the embeddings or the clustering algorithm, I just describe the task in plain English and let the model apply its understanding of the app market. The structured output approach worked well — using Pydantic + the `parse` API means the model never returns something I can't directly join back to the DataFrame. No regex, no fragile JSON parsing.

The same logic would apply to KMeans clusters. I left it as an extension rather than implementing it because the KNN clusters were already a better starting point.

### Why `BAAI/bge-base-en`?

I needed a sentence embedding model that's good at capturing semantic similarity between short, feature-list style texts. `BAAI/bge-base-en` consistently shows up at the top of the MTEB leaderboard for retrieval and semantic similarity tasks, which is exactly the use case here. It produces 768-dimensional vectors, which is a reasonable size — expressive enough to capture nuance, compact enough to run KNN efficiently on 4 000 apps without needing a GPU or waiting hours. It also supports L2 normalisation natively, which makes cosine similarity a simple dot product downstream.

I considered using larger models but the base variant already gave me clearly meaningful clusters on the first run, so I didn't see a reason to pay the speed cost of something bigger.

### Why `gpt-4o-mini`?

This was a cost and quality trade-off. I'm sending ~700+ cluster prompts, each with several app descriptions. Using `gpt-4` or `gpt-4o` for all of them would be expensive and slow. `gpt-4o-mini` is fast, cheap, and — for structured classification tasks like "group these apps by who they compete with" — more than capable. The quality of its sub-niche labels was consistently good in my testing: the names were sensible, the descriptions were accurate, and it reliably followed the structured output schema without hallucinating extra fields or dropping apps.

If I were running this in production on the full dataset and wanted higher confidence on edge cases, I'd probably use `gpt-4o` for the largest, most ambiguous clusters and `gpt-4o-mini` for the smaller ones. But for this task, `gpt-4o-mini` across the board was the right call.

---

## Pipeline Overview

The pipeline runs in two stages:

```
Raw app data
    │
    ▼
[Stage 1] Embedding-based clustering
    │   Encode app features → cluster similar apps
    │
    ▼
[Stage 2] LLM sub-niche refinement
        Validate clusters → split into direct-competitor groups
        → app_sub_niches.csv
```

**Why two stages?**
Embedding similarity alone is not sufficient to identify direct competitors. Two apps can be semantically close (e.g. both use "AI image generation") yet serve completely different user needs (portrait photos vs. logo design). The LLM acts as a second filter that applies market analysis reasoning to make this distinction.

---

## Approach 1 — KMeans (`kmeans.ipynb`)

Cluster apps into `k = 700` groups by running KMeans on L2-normalised sentence embeddings.

**How it works:**
1. Encode the `features` column with `BAAI/bge-base-en`
2. Run `KMeans(n_clusters=700)` on the normalised embedding matrix
3. Every app is assigned to exactly one cluster

**Pros:**
- Fast and simple
- Guarantees every app gets a cluster — no noise/outlier label
- Deterministic with a fixed random seed

**Cons:**
- Requires pre-specifying `k`; the right value is not obvious
- Assumes clusters are convex and roughly equal in size — a poor fit for a long-tail app market where some niches have 2 apps and others have 50
- KMeans minimises Euclidean distance; cosine similarity (more appropriate for embeddings) is only approximated

**LLM refinement potential:** the same `llm_validation.ipynb` pipeline used for NN clusters could be applied directly to KMeans clusters — each cluster is sent to the LLM to be split into sub-niches of direct competitors. This is left as a straightforward extension.

---

## Approach 2 — KNN Graph + Connected Components (`NearestNeighbors.ipynb`) 

Build a similarity graph where apps are nodes and edges connect apps whose cosine similarity exceeds a threshold. Clusters emerge as the connected components of this graph.

**How it works:**
1. For each app, find its `k = 2` nearest neighbours (cosine distance)
2. Add an undirected edge between two apps if `cosine_similarity ≥ 0.8`
3. Extract connected components — each component is one cluster
4. Apps with no edges (isolated nodes) receive label `−1` (noise)

**Pros:**
- No need to specify the number of clusters
- Threshold is directly interpretable: cluster members are guaranteed to be ≥ 80% similar to at least one neighbour
- Naturally handles non-convex cluster shapes and variable cluster sizes

**Cons:**
- Sensitive to the similarity threshold; a small change can merge or split clusters significantly
- Chain effect: A→B and B→C are connected even if A and C are not directly similar, which can create unexpectedly large clusters
- Apps below the threshold become noise and are excluded from downstream analysis

The KNN graph clusters proved to be good enough as a first-stage result on their own — apps within each component are genuinely similar and intuitively grouped. However, "similar" is still not the same as "direct competitor": a single component can contain apps that share the same technology or domain but address different user needs. The LLM validation step is therefore not a correction of poor clustering, but a semantic refinement of already solid groups — splitting each component into tighter sub-niches where every app truly competes for the same user.

---

## Approach 3 — BERTopic (`topic_modeling.ipynb`)

Use BERTopic — a topic modelling framework combining UMAP dimensionality reduction, HDBSCAN density clustering, and c-TF-IDF keyword extraction.

**How it works:**
1. Reduce 768-dim embeddings to 2D with UMAP (`n_neighbors=5`, `min_dist=0.0`)
2. Cluster the 2D projection with HDBSCAN (`min_cluster_size=2`)
3. Represent each cluster with its top c-TF-IDF keywords to produce auto-labelled topics

**Pros:**
- Fully unsupervised — no `k` to tune
- Produces human-readable keyword labels for each topic automatically
- Handles outliers explicitly with the `−1` noise label

**Cons:**
- HDBSCAN assigned ~695 apps (~16%) to topic `−1` (outliers) — a large fraction of the dataset that effectively drops from the analysis
- Clustering quality depends heavily on the 2D UMAP projection; information is inevitably lost in such aggressive dimensionality reduction (768 → 2)
- The auto-generated keyword labels describe vocabulary patterns, not competitive market structure — they require further interpretation
- Topics tend to be broader than desired for sub-niche discovery (e.g. a single "AI art" topic contains dozens of apps with distinct use cases)

The high outlier rate (16%) was the decisive factor. Any app labelled `−1` by HDBSCAN is excluded from the sub-niche analysis entirely. In a competitive intelligence context, dropping 1-in-6 apps means the final output has significant blind spots. The KNN graph approach produces far fewer excluded apps while giving tighter, more actionable clusters.

---

## Stage 2 — LLM Sub-niche Refinement (`llm_validation.ipynb`)

For each embedding cluster, send all apps (with `trackName`, `overview`, and `features`) to `gpt-4o-mini` and ask it to split the cluster into groups of **direct competitors**.

**Output per sub-niche:**
- `sub_niche` — short name (e.g. "AI logo maker")
- `sub_niche_description` — 2–3 sentences describing the problem it solves, target user, and what distinguishes it from adjacent categories
- `competitors` — list of `trackName` values

**Why LLM as a second step:**
Embedding models capture semantic similarity but lack market knowledge. The LLM brings domain reasoning: it understands that "Duolingo" and "a meditation app" might both appear in a "daily habit" cluster yet are not competitors. Structured output (Pydantic + OpenAI `parse` API) guarantees the response schema, making the output directly joinable back to the DataFrame.

**Current scope:** clusters 0–50 are processed. Extend `valid_clusters` in `llm_validation.ipynb` to run over all 723 clusters.

---

## Reproducing the Results

```bash
pip install -r requirements.txt

# Add your OpenAI API key to .env:
# OPENAI_API_KEY=sk-...

# Stage 1: run NearestNeighbors.ipynb top-to-bottom
# Stage 2: run llm_validation.ipynb top-to-bottom
# Final output: app_sub_niches.csv
```
