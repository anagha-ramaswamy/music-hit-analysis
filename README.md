# The Anatomy of a Hit

**DATASCI 112 Final Project**

## Research Question
Are hit songs converging — sounding and saying the same thing — and if so, when did it start, and did streaming reverse it?

## Data Sources & Collection

### 1. Billboard Hot 100 (billboard.py)
- Year-end charts 1970-2023 via `billboard.py` API
- Weekly chart backfill for 1992-2006 where year-end charts unavailable
- ~3,300+ unique songs with chart position and year

### 2. Audio Features (YouTube + librosa)
- `yt-dlp` searches YouTube for each song and downloads 30 seconds of audio
- `librosa` extracts: tempo, energy, loudness, danceability, acousticness, speechiness, instrumentalness, valence
- Checkpoint-based collection with resume support

### 3. Lyrics (Genius API + lyricsgenius)
- Lyrics scraped via Genius API with fuzzy title/artist matching
- NLP features computed: lexical diversity, word count, sentiment, avg word length
- Checkpoint-based collection with resume support

## Pipeline

Run `complete_analysis.ipynb` in order:

| Step | Script | Output |
|------|--------|--------|
| 0 | `Phase_0_Billboard_Data.ipynb` | `songs.csv` |
| 1 | `spotify_audio_features.py` | `audio_raw.csv` → `audio_clean.csv` |
| 2 | `genius_lyrics_scraper.py` | `lyrics_clean.csv` |
| 3 | `audio_analysis.py` | Audio visualizations |
| 4 | `nlp_analysis.py` | NLP visualizations |
| 5 | `merge_datasets.py` | `merged.csv` |
| 6 | `final_analysis.py` | ML models + summary poster |

## Techniques Used

### Audio Analysis
- KDE distribution plots by decade
- Feature trend lines over time
- PCA with convex hulls (cluster area = convergence metric)
- KMeans clustering
- Audio diversity (std dev) by decade

### NLP Analysis
- Lexical diversity trends with quadratic trend fit
- TextBlob sentiment analysis
- LDA topic modeling (lyrical theme diversity)
- Word2Vec semantic drift across decades
- TF-IDF + UMAP lyrical clustering

### Machine Learning
- **Decade classification** (Random Forest, KNN, Logistic Regression)
  - Predicts which decade a song is from based on audio + lyrics features
  - Confusion matrix reveals which eras sound most similar
  - Per-decade accuracy directly measures sonic distinctiveness
- **SHAP values** for feature importance interpretability

## Key Findings

1. **U-shaped lexical diversity**: Hit lyrics got simpler through the 2000s, then recovered in the streaming era (quadratic fit R²=0.68)
2. **Sonic convergence then divergence**: PCA cluster areas shrank 1970s→2000s, then expanded in the 2010s
3. **Lyrical themes diversified**: LDA topic modeling shows increasing thematic heterogeneity across decades
4. **Sentiment darkened**: Hit songs trended toward more negative sentiment over time
5. **Streaming disrupted the formula**: The 2010s reversal across all metrics points to streaming-era genre fragmentation

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install system dependencies (macOS)
brew install ffmpeg deno
```

## API Credentials Required

- **Genius API**: Set `GENIUS_ACCESS_TOKEN` in notebook cell 9
- Spotify API no longer required (replaced by YouTube + librosa)

## Project Structure

```
music-hit-analysis/
├── complete_analysis.ipynb          # Main pipeline notebook
├── Phase_0_Billboard_Data.ipynb     # Billboard data collection
├── spotify_audio_features.py        # YouTube + librosa audio extraction
├── genius_lyrics_scraper.py         # Genius lyrics scraper
├── audio_analysis.py                # Audio feature analysis & visualization
├── nlp_analysis.py                  # NLP analysis & visualization
├── merge_datasets.py                # Dataset merging
├── final_analysis.py                # ML models + summary visualization
├── requirements.txt
├── songs.csv                        # Billboard chart data
├── audio_raw.csv                    # Raw audio features
├── audio_clean.csv                  # Cleaned audio features
├── lyrics/                          # Individual lyrics .txt files
├── lyrics_clean.csv                 # Processed lyrics with NLP features
└── merged.csv                       # Combined dataset for ML
```
