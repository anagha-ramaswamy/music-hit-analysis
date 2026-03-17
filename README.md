# The Anatomy of a Hit

**DATASCI 112 Final Project — Stanford University**
Anagha Ramaswamy & Abraham Yeung

## Research Question

Are Billboard Hot 100 hits converging — sounding and saying the same thing — and when did it start? We examine audio and lyrics as independent channels and find they moved together for decades, then **decoupled around 2016**: lyrics recovered to 1970s diversity while audio production continues to homogenize.

---

## Data

| Source | Method | Output |
|--------|--------|--------|
| Billboard Hot 100 | `billboard.py` year-end charts; weekly backfill for 1992–2006 | `songs.csv` (4,801 songs, 1970–2023) |
| Audio features | `yt-dlp` (30-sec YouTube clips) + `librosa` feature extraction | `audio_clean.csv` (4,801 songs) |
| Lyrics | Genius API + `lyricsgenius`, BeautifulSoup fallback | `lyrics_clean.csv` (4,554 songs after removing bad scrapes) |

Songs per decade: 796 / 866 / 984 / 913 / 888 / 354 (1970s–2020s; 2020s is 2020–2023 only).

---

## Pipeline

Run scripts in order:

```
Phase_0_Billboard_Data.ipynb   →  songs.csv
spotify_audio_features.py      →  audio_raw.csv → audio_clean.csv
genius_lyrics_scraper.py       →  lyrics_clean.csv
audio_analysis.py              →  audio_feature_trends.png
nlp_analysis.py                →  temporal_patterns_analysis.png, lda_topic_analysis.png
merge_datasets.py              →  merged.csv, ml_features.csv
final_analysis.py              →  decade_similarity_heatmap.png, decade_dendrogram.png
```

---

## Analysis

### Audio (`audio_analysis.py`)
8 features extracted via `librosa`: tempo, energy, loudness, danceability, acousticness, speechiness, instrumentalness, valence.

- **Feature trends over time** — mean ± 1 s.d. per year for all 8 features (`audio_feature_trends.png`)

### NLP (`nlp_analysis.py`)
Bad scrapes removed with a word-count filter (>800 words flagged). Custom stopword list shared across all vectorizers.

- **Lexical diversity** — unique/total words per song, trends by year and decade
- **LZ77 compressibility** — structural repetition metric from Interiano et al. (2018); computed per song
- **Sentiment** — TextBlob polarity (−1 to +1)
- **LDA topic modeling** — `CountVectorizer` + LDA (k=6) on `lyrics_clean`; per-decade topic proportion heatmap (`lda_topic_analysis.png`)
- **Statistical tests** — Shapiro-Wilk confirmed non-Gaussian distributions; Kruskal-Wallis + pairwise Mann-Whitney U + Bonferroni correction on all decade comparisons
- **Lexical diversity + sentiment trends** — combined plot with quadratic (LD) and linear (sentiment) fits (`temporal_patterns_analysis.png`)

### Merged & ML (`merge_datasets.py`, `final_analysis.py`)
Features: 8 audio + lexical diversity + compressibility + sentiment polarity = 11 features total.

- **Cosine similarity heatmap** — standardized decade audio centroids, 6×6 similarity matrix (`decade_similarity_heatmap.png`)
- **Hierarchical clustering dendrogram** — Ward linkage on 6 decade centroids; confirms pre/post-digital era split (`decade_dendrogram.png`)
- **Decade classification** — Random Forest, KNN (k=7), Logistic Regression; low accuracy (~34%) is the finding (2× random baseline; heavy decade overlap = convergence evidence)
- **K-Means (k=6) + ARI** — clusters cut across decade boundaries (ARI = 0.013), confirming individual songs don't separate cleanly by era
- **SHAP values** — averaged across all 6 classes; year and loudness are the dominant era predictors

---

## Key Findings

1. **Audio and lyrics decoupled ~2016.** They converged together 1970s–2010s, then diverged: lyrics recovered while audio homogenized further.
2. **Lexical diversity U-shape** — declined 0.383 (1970s) → 0.342 (2010s trough), recovered to 0.381 (2020s). Kruskal-Wallis H=99.89, p=5.56×10⁻²⁰. 2020s not significantly different from 1970s (p=1.0, Bonferroni).
3. **Lyrical recovery is genre-driven** — LDA shows the 2020s draw from a wider thematic pool (hip-hop 8%→24%, Latin 1%→14%) rather than individual songs becoming more complex.
4. **Sonic convergence accelerated** — 2020s convex hull area (41.9) is the smallest on record. Cosine similarity between 2010s and 2020s centroids = +0.92.
5. **1980s and 2020s are acoustic opposites** — cosine similarity = −0.99, driven by loudness war (−23.6→−18.7 dB), rising speechiness, declining acousticness.
6. **Sentiment never recovered** — monotonic decline 0.165 (1970s) → 0.080 (2020s); operates independently of vocabulary diversity.
7. **Pre/post-digital era split** — hierarchical clustering separates 1970s–1990s from 2000s–2020s; 1990s is the hinge decade between the two regimes.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
brew install ffmpeg  # macOS, required for yt-dlp audio extraction
```

**API credentials:**
- `GENIUS_ACCESS_TOKEN` — set in `genius_lyrics_scraper.py` before running

---

## Project Structure

```
music-hit-analysis/
├── Phase_0_Billboard_Data.ipynb     # Billboard data collection
├── spotify_audio_features.py        # yt-dlp + librosa audio extraction
├── genius_lyrics_scraper.py         # Genius lyrics scraper
├── audio_analysis.py                # Audio feature trends
├── nlp_analysis.py                  # Lexical diversity, sentiment, LDA, compressibility
├── merge_datasets.py                # Merge audio + lyrics; build ML features
├── final_analysis.py                # Cosine similarity, dendrogram, ML models, SHAP
├── requirements.txt
├── poster.tex                       # Conference poster (Beamer/Gemini theme)
├── songs.csv                        # Billboard chart data (4,801 songs)
├── audio_clean.csv                  # Cleaned audio features
├── lyrics_clean.csv                 # Processed lyrics with NLP features
├── merged.csv                       # Combined audio + lyrics dataset
└── ml_features.csv                  # Scaled features for ML models
```
