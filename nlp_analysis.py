"""
NLP Lyrics Analysis
Analyzes lyrics for linguistic patterns, sentiment, and topic modeling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NLPAnalyzer:
    def __init__(self, lyrics_file='lyrics_clean.csv'):
        """Initialize with lyrics data"""
        self.df = pd.read_csv(lyrics_file)
        self.df = self.df.dropna(subset=['lyrics_clean'])
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analysis"""
        # Add decade column
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df['decade'] = self.df['decade'].astype(str) + 's'

        # Drop bad Genius scrapes: real songs are almost never over 800 words.
        before = len(self.df)
        self.df = self.df[self.df['word_count'] <= 800].copy()
        dropped = before - len(self.df)
        if dropped > 0:
            print(f"Dropped {dropped} likely bad scrapes (word_count > 800)")

        # Merge chart position from songs.csv so we can filter by top-N
        if 'chart_position' not in self.df.columns:
            try:
                songs = pd.read_csv('songs.csv')[['title', 'artist', 'year', 'chart_position']]
                self.df = self.df.merge(songs, on=['title', 'artist', 'year'], how='left')
            except FileNotFoundError:
                print("Warning: songs.csv not found — chart_position filtering unavailable")

        print(f"Loaded {len(self.df)} songs with lyrics")
        print(f"Decades covered: {', '.join(sorted(self.df['decade'].unique()))}")

    @property
    def lyric_stopwords(self):
        """Shared stopword list for all vectorizers: sklearn English + contraction
        artifacts + lyric filler + metadata leakage + Spanish filler."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return list(set(ENGLISH_STOP_WORDS) | {
            # Contraction artifacts (apostrophe stripped)
            'll', 've', 're', 'd', 't', 's', 'ain', 'don', 'won', 'can',
            'didn', 'doesn', 'isn', 'wasn', 'weren', 'wouldn', 'couldn', 'shouldn',
            # Lyric filler words
            'oh', 'ooh', 'ah', 'yeah', 'ya', 'na', 'la', 'da', 'hey', 'uh',
            'wanna', 'gonna', 'gotta', 'whoa', 'mmm', 'hmm', 'ha', 'baby',
            'like', 'just', 'got', 'know', 'let', 'say', 'said', 'come',
            'get', 'go', 'going', 'cause', 'cuz', 'make', 'take', 'want',
            'need', 'feel', 'way', 'right', 'left', 'little', 'bout',
            # Metadata leakage
            'feat', 'ft', 'remix', 'edit', 'version', 'radio',
            # Spanish filler
            'que', 'tu', 'mi', 'te', 'yo', 'es', 'de', 'en', 'el', 'lo',
        })

    def analyze_sentiment(self):
        """Compute TextBlob sentiment for all lyrics and save to lyrics_clean.csv.
        This must run before merge_datasets.py so sentiment columns are in the merged data."""
        sentiments = []

        for lyrics in tqdm(self.df['lyrics_clean'], desc="Analyzing sentiment"):
            blob = TextBlob(lyrics)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            sentiments.append({
                'vader_compound': polarity,  # TextBlob polarity stored under vader_compound name
                'vader_positive': max(polarity, 0),
                'vader_negative': abs(min(polarity, 0)),
                'vader_neutral': 1 - abs(polarity),
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity
            })

        sentiment_df = pd.DataFrame(sentiments)
        self.df = pd.concat([self.df.reset_index(drop=True), sentiment_df], axis=1)
        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='last')]

        yearly_sentiment = self.df.groupby('year')[['vader_compound', 'textblob_polarity']].mean()

        # Save enriched dataframe back so merge_datasets.py can access sentiment columns
        self.df.to_csv('lyrics_clean.csv', index=False)
        print("Saved sentiment columns to lyrics_clean.csv")

        return yearly_sentiment

    def analyze_lda_topics(self, n_topics=6):
        """LDA topic modeling to show how lyrical themes shift by decade → lda_topic_analysis.png"""
        # Use count vectorizer (LDA needs raw counts, not TF-IDF)
        count_vec = CountVectorizer(max_features=3000, stop_words=self.lyric_stopwords,
                                    min_df=5, max_df=0.8)
        count_matrix = count_vec.fit_transform(self.df['lyrics_clean'])
        feature_names = count_vec.get_feature_names_out()

        # Fit LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                        max_iter=20, learning_method='online')
        doc_topics = lda.fit_transform(count_matrix)

        # Get top words per topic
        topic_labels = []
        for i, topic in enumerate(lda.components_):
            top_words = [feature_names[j] for j in topic.argsort()[-8:][::-1]]
            topic_labels.append(f"Topic {i+1}: {', '.join(top_words[:4])}")

        # Add dominant topic to df
        self.df['dominant_topic'] = doc_topics.argmax(axis=1)

        # Plot: topic prevalence by decade
        topic_by_decade = pd.DataFrame(doc_topics, columns=[f'Topic {i+1}' for i in range(n_topics)])
        topic_by_decade['decade'] = self.df['decade'].values
        decade_topics = topic_by_decade.groupby('decade').mean()

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Stacked bar: topic mix per decade
        decade_topics.plot(kind='bar', stacked=True, ax=axes[0],
                           colormap='tab10', alpha=0.85)
        axes[0].set_xlabel('Decade', fontsize=12)
        axes[0].set_ylabel('Mean Topic Proportion', fontsize=12)
        axes[0].set_title('Lyrical Theme Mix by Decade', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].tick_params(axis='x', rotation=45)

        # Heatmap of topic proportions
        sns.heatmap(decade_topics.T, cmap='YlOrRd', annot=True, fmt='.2f',
                    ax=axes[1], cbar_kws={'label': 'Mean Proportion'})
        axes[1].set_title('Topic Heatmap by Decade', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Topic')

        plt.tight_layout()
        plt.savefig('lda_topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nTop words per topic:")
        for label in topic_labels:
            print(f"  {label}")

        return lda, doc_topics, topic_labels

    def test_decade_significance(self):
        """Kruskal-Wallis + pairwise Mann-Whitney U + Bonferroni correction for lexical diversity.
        Produces the H and p statistics cited in the poster."""
        from scipy import stats
        from itertools import combinations

        decades = sorted(self.df['decade'].unique())
        groups = [self.df[self.df['decade'] == d]['lexical_diversity'].dropna().values
                  for d in decades]

        # Kruskal-Wallis across all decades
        kw_stat, kw_p = stats.kruskal(*groups)

        # Pairwise Mann-Whitney U with Bonferroni correction
        pairs = list(combinations(range(len(decades)), 2))
        n_comparisons = len(pairs)
        pairwise = {}
        for i, j in pairs:
            u_stat, p_val = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            p_bonf = min(p_val * n_comparisons, 1.0)
            pairwise[(decades[i], decades[j])] = {
                'u_stat': u_stat,
                'p_raw': p_val,
                'p_bonferroni': p_bonf,
                'significant': p_bonf < 0.05
            }

        print(f"\n{'='*55}")
        print("Lexical Diversity — Statistical Significance")
        print(f"{'='*55}")
        print(f"Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4e} "
              f"({'SIGNIFICANT' if kw_p < 0.05 else 'not significant'})")
        print(f"\nPairwise Mann-Whitney U (Bonferroni corrected, n={n_comparisons}):")
        print(f"{'Pair':<25} {'p (raw)':>10} {'p (Bonf.)':>12} {'Sig?':>6}")
        print("-" * 55)
        for (d1, d2), res in pairwise.items():
            sig = ('***' if res['p_bonferroni'] < 0.001 else
                   '**' if res['p_bonferroni'] < 0.01 else
                   '*' if res['significant'] else '')
            print(f"{d1} vs {d2:<10} {res['p_raw']:>10.4f} {res['p_bonferroni']:>12.4f} {sig:>6}")

        return {'kruskal_stat': kw_stat, 'kruskal_p': kw_p, 'pairwise': pairwise}

    @staticmethod
    def _lz77_compressibility(text, window_size=32768, min_match=4, max_match=258):
        """LZ77 compressibility per Interiano et al. (2018) methodology.

        compsize(S) = |S| - sum(L - 3) for all matches with L >= 4
        compressibility = log(|S| / compsize(S))

        Higher value = more repetitive/compressible lyrics.
        """
        import math
        n = len(text)
        if n < min_match:
            return 0.0

        total_savings = 0
        i = 0

        while i < n:
            best_len = 0

            if i + min_match <= n:
                win_start = max(0, i - window_size)
                seed = text[i:i + min_match]
                search_from = win_start

                while True:
                    pos = text.find(seed, search_from, i)
                    if pos == -1:
                        break
                    dist = i - pos
                    length = min_match
                    while i + length < n and length < max_match:
                        if text[pos + (length % dist)] == text[i + length]:
                            length += 1
                        else:
                            break
                    if length > best_len:
                        best_len = length
                    search_from = pos + 1

            if best_len >= min_match:
                total_savings += best_len - 3
                i += best_len
            else:
                i += 1

        comp_size = max(n - total_savings, 1)
        return math.log(n / comp_size)

    def analyze_compressibility(self):
        """Compute LZ77 compressibility for each song and save to lyrics_clean.csv.
        Must run before merge_datasets.py so compressibility is in the merged data."""
        if 'compressibility' not in self.df.columns:
            print("Computing LZ77 compressibility (this takes a few minutes)...")
            tqdm.pandas(desc="LZ77")
            self.df['compressibility'] = self.df['lyrics_clean'].progress_apply(
                self._lz77_compressibility
            )
            self.df.to_csv('lyrics_clean.csv', index=False)
            print("Saved compressibility to lyrics_clean.csv")

        decade_comp = self.df.groupby('decade')['compressibility'].agg(['mean', 'std', 'count'])
        print("\nLZ77 Compressibility by Decade (higher = more repetitive):")
        print(decade_comp.round(3).to_string())

        return decade_comp

    def run_full_analysis(self):
        """Run complete NLP analysis pipeline"""
        print("Starting NLP Lyrics Analysis...")
        print("=" * 50)

        print("\n1. Computing and saving sentiment...")
        self.analyze_sentiment()

        print("\n2. Computing and saving LZ77 compressibility...")
        self.analyze_compressibility()

        print("\n3. Running LDA topic modeling...")
        self.analyze_lda_topics()

        print("\n4. Running significance tests...")
        self.test_decade_significance()

        print("\nAnalysis complete! Check lda_topic_analysis.png")


def main():
    if not os.path.exists('lyrics_clean.csv'):
        print("Error: lyrics_clean.csv not found.")
        return

    analyzer = NLPAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
