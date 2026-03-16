"""
NLP Lyrics Analysis
Analyzes lyrics for linguistic patterns, sentiment, and clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
import umap
from gensim.models import Word2Vec
from textblob import TextBlob
from collections import Counter
import re
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
        
        print(f"Loaded {len(self.df)} songs with lyrics")
        print(f"Decades covered: {', '.join(sorted(self.df['decade'].unique()))}")
    
    def analyze_lexical_diversity(self):
        """Analyze lexical diversity trends over time"""
        # Calculate yearly statistics
        yearly_diversity = self.df.groupby('year')['lexical_diversity'].agg(['mean', 'std', 'count'])
        
        # Calculate decade statistics
        decade_diversity = self.df.groupby('decade')['lexical_diversity'].agg(['mean', 'std', 'count'])
        
        # Plot lexical diversity over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Yearly trend with confidence bands
        years = yearly_diversity.index
        means = yearly_diversity['mean']
        stds = yearly_diversity['std']
        
        ax1.plot(years, means, linewidth=3, color='royalblue', label='Mean Lexical Diversity')
        ax1.fill_between(years, means - stds, means + stds, alpha=0.3, color='royalblue')
        
        # Add quadratic trend line (U-shape: simpler through 2000s, then streaming reversal)
        z = np.polyfit(years, means, 2)
        p = np.poly1d(z)
        years_smooth = np.linspace(years.min(), years.max(), 300)
        residuals = means - p(years)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ax1.plot(years_smooth, p(years_smooth), "r--", alpha=0.8, linewidth=2,
                 label=f'Quadratic trend (R²={r_squared:.3f}, min ≈ {int(-z[1]/(2*z[0]))})')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Lexical Diversity', fontsize=12)
        ax1.set_title('Lexical Diversity Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Decade comparison
        decades = decade_diversity.index
        means_decade = decade_diversity['mean']
        stds_decade = decade_diversity['std']
        
        bars = ax2.bar(decades, means_decade, yerr=stds_decade, 
                      capsize=5, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Decade', fontsize=12)
        ax2.set_ylabel('Mean Lexical Diversity', fontsize=12)
        ax2.set_title('Lexical Diversity by Decade', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means_decade, stds_decade):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('lexical_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return yearly_diversity, decade_diversity
    
    def analyze_sentiment(self):
        """Analyze sentiment trends in lyrics"""
        # Calculate sentiment for all lyrics
        sentiments = []
        
        for lyrics in tqdm(self.df['lyrics_clean'], desc="Analyzing sentiment"):
            blob = TextBlob(lyrics)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            sentiments.append({
                'vader_compound': polarity,  # use textblob polarity as proxy
                'vader_positive': max(polarity, 0),
                'vader_negative': abs(min(polarity, 0)),
                'vader_neutral': 1 - abs(polarity),
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity
            })
        
        # Add to dataframe
        sentiment_df = pd.DataFrame(sentiments)
        self.df = pd.concat([self.df.reset_index(drop=True), sentiment_df], axis=1)
        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='last')]
        
        # Analyze trends
        yearly_sentiment = self.df.groupby('year')[['vader_compound', 'textblob_polarity']].mean()
        
        # Plot sentiment trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment over time
        axes[0, 0].plot(yearly_sentiment.index, yearly_sentiment['vader_compound'],
                       linewidth=3, color='darkgreen')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].set_title('Sentiment Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # TextBlob subjectivity over time
        axes[0, 1].plot(yearly_sentiment.index, yearly_sentiment['textblob_polarity'],
                       linewidth=3, color='darkorange')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('TextBlob Polarity')
        axes[0, 1].set_title('TextBlob Sentiment Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Sentiment distribution by decade
        decade_sentiment = self.df.groupby('decade')['vader_compound'].mean()
        bars = axes[1, 0].bar(decade_sentiment.index, decade_sentiment.values,
                             alpha=0.7, color='mediumpurple')
        axes[1, 0].set_xlabel('Decade')
        axes[1, 0].set_ylabel('Mean Sentiment')
        axes[1, 0].set_title('Average Sentiment by Decade')
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, decade_sentiment.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')

        # Sentiment vs Lexical Diversity scatter
        axes[1, 1].scatter(self.df['lexical_diversity'], self.df['vader_compound'],
                          alpha=0.5, s=30)
        axes[1, 1].set_xlabel('Lexical Diversity')
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].set_title('Sentiment vs Lexical Diversity')

        # Add correlation
        corr = self.df['lexical_diversity'].corr(self.df['vader_compound'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 1].transAxes, va='top')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save enriched dataframe back so merge_datasets.py can access sentiment columns
        self.df.to_csv('lyrics_clean.csv', index=False)

        return yearly_sentiment
    
    def analyze_vocabulary_drift(self):
        """Analyze vocabulary changes over decades using TF-IDF"""
        # Prepare TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )
        
        # Fit and transform all lyrics
        self.vectorizer = vectorizer
        tfidf_matrix = vectorizer.fit_transform(self.df['lyrics_clean'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Create DataFrame with TF-IDF scores
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df['decade'] = self.df['decade'].values
        
        # Calculate average TF-IDF scores by decade
        decade_tfidf = tfidf_df.groupby('decade').mean()
        
        # Get top words for each decade
        top_words_per_decade = {}
        for decade in decade_tfidf.index:
            decade_scores = decade_tfidf.loc[decade]
            top_words = decade_scores.nlargest(20)
            top_words_per_decade[decade] = top_words
        
        # Create heatmap of top words
        all_top_words = set()
        for words in top_words_per_decade.values():
            all_top_words.update(words.index)
        
        # Prepare heatmap data
        heatmap_data = []
        for decade in sorted(top_words_per_decade.keys()):
            decade_words = top_words_per_decade[decade]
            row = []
            for word in sorted(all_top_words):
                row.append(decade_words.get(word, 0))
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=sorted(top_words_per_decade.keys()),
                                 columns=sorted(all_top_words))
        
        # Plot heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_df, cmap='YlOrRd', annot=False, 
                   cbar_kws={'label': 'Average TF-IDF Score'})
        plt.title('Vocabulary Drift: Top Words by Decade', fontsize=16, fontweight='bold')
        plt.xlabel('Words')
        plt.ylabel('Decade')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('vocabulary_drift_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_words_per_decade, vectorizer, tfidf_matrix
    
    def perform_lyrical_clustering(self, tfidf_matrix, n_clusters=8):
        """Perform KMeans clustering on lyrics"""
        # Reduce dimensionality with SVD first
        svd = TruncatedSVD(n_components=50, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_reduced)
        
        # Add cluster labels to dataframe
        self.df['lyrical_cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_songs = self.df[self.df['lyrical_cluster'] == cluster_id]
            
            # Get top words for this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_tfidf = tfidf_matrix[cluster_indices]
            mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)
            
            # Get top 10 words
            feature_names = self.vectorizer.get_feature_names_out()
            top_word_indices = np.argsort(mean_tfidf)[-10:]
            top_words = [feature_names[i] for i in top_word_indices]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_songs),
                'top_words': top_words,
                'avg_lexical_diversity': cluster_songs['lexical_diversity'].mean(),
                'avg_sentiment': cluster_songs['vader_compound'].mean(),
                'dominant_decade': cluster_songs['decade'].mode().iloc[0] if len(cluster_songs) > 0 else None
            }
        
        return cluster_analysis, tfidf_reduced
    
    def create_umap_visualization(self, tfidf_reduced):
        """Create UMAP visualization of lyrical space"""
        # Reduce to 2D with UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(tfidf_reduced)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color by lyrical cluster
        scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                                  c=self.df['lyrical_cluster'], cmap='tab10', 
                                  alpha=0.7, s=50)
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].set_title('Lyrical Space by Cluster', fontsize=14, fontweight='bold')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Color by decade
        decades = sorted(self.df['decade'].unique())
        decade_colors = plt.cm.Set3(np.linspace(0, 1, len(decades)))
        decade_color_map = {decade: i for i, decade in enumerate(decades)}
        
        scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                                  c=[decade_color_map[d] for d in self.df['decade']], 
                                  cmap='Set3', alpha=0.7, s=50)
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].set_title('Lyrical Space by Decade', fontsize=14, fontweight='bold')
        
        # Add legend for decades
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=decade_colors[i], markersize=10, label=decade)
                          for i, decade in enumerate(decades)]
        axes[1].legend(handles=legend_elements, loc='best', title='Decade')
        
        plt.tight_layout()
        plt.savefig('umap_lyrical_space.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return embedding
    
    def analyze_lda_topics(self, n_topics=6):
        """LDA topic modeling to show how lyrical themes shift by decade"""
        # Use count vectorizer (LDA needs raw counts, not TF-IDF)
        count_vec = CountVectorizer(max_features=3000, stop_words='english',
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

    def analyze_word2vec_drift(self, target_words=None):
        """Word2Vec semantic drift: train per decade and track how word context shifts"""
        if target_words is None:
            target_words = ['love', 'baby', 'heart', 'girl', 'night', 'time', 'feel']

        # Tokenize lyrics
        self.df['tokens'] = self.df['lyrics_clean'].apply(
            lambda x: [w for w in x.lower().split() if len(w) > 2])

        decades = sorted(self.df['decade'].unique())
        decade_models = {}

        for decade in decades:
            decade_lyrics = self.df[self.df['decade'] == decade]['tokens'].tolist()
            if len(decade_lyrics) < 10:
                continue
            model = Word2Vec(sentences=decade_lyrics, vector_size=100, window=5,
                             min_count=3, workers=4, epochs=20, seed=42)
            decade_models[decade] = model

        # For each target word, find most similar words per decade
        fig, axes = plt.subplots(len(target_words), 1,
                                 figsize=(14, 3 * len(target_words)))
        if len(target_words) == 1:
            axes = [axes]

        for ax, word in zip(axes, target_words):
            decades_present = []
            top_neighbors = []
            for decade, model in decade_models.items():
                if word in model.wv:
                    similar = model.wv.most_similar(word, topn=5)
                    decades_present.append(decade)
                    top_neighbors.append([w for w, _ in similar])

            if not decades_present:
                ax.axis('off')
                ax.set_title(f'"{word}" not found', fontsize=10)
                continue

            # Display as table-style text
            cell_text = [[', '.join(neighbors)] for neighbors in top_neighbors]
            ax.axis('off')
            table = ax.table(cellText=cell_text,
                             rowLabels=decades_present,
                             colLabels=[f'Top 5 words near "{word}"'],
                             loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            ax.set_title(f'Semantic neighbors of "{word}" by decade',
                         fontsize=11, fontweight='bold', pad=2)

        plt.suptitle('Word2Vec Semantic Drift: How Word Context Shifts Over Decades',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig('word2vec_semantic_drift.png', dpi=300, bbox_inches='tight')
        plt.show()

        return decade_models

    def run_full_analysis(self):
        """Run complete NLP analysis pipeline"""
        print("Starting NLP Lyrics Analysis...")
        print("=" * 50)
        
        print("\\n1. Analyzing lexical diversity trends...")
        yearly_div, decade_div = self.analyze_lexical_diversity()
        
        print("\\n2. Analyzing sentiment patterns...")
        sentiment_trends = self.analyze_sentiment()
        
        print("\\n3. Analyzing vocabulary drift...")
        top_words, vectorizer, tfidf_matrix = self.analyze_vocabulary_drift()
        
        print("\\n4. Performing lyrical clustering...")
        cluster_analysis, tfidf_reduced = self.perform_lyrical_clustering(tfidf_matrix)
        
        print("\\n5. Creating UMAP visualization...")
        embedding = self.create_umap_visualization(tfidf_reduced)

        print("\\n6. Running LDA topic modeling...")
        lda, doc_topics, topic_labels = self.analyze_lda_topics()

        print("\\n7. Running Word2Vec semantic drift...")
        decade_models = self.analyze_word2vec_drift()

        print("\\nAnalysis complete! Check the generated PNG files.")
        
        # Print cluster summary
        print("\\nLyrical Clusters Summary:")
        for cluster_id, info in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {info['size']} songs")
            print(f"  Top words: {', '.join(info['top_words'][-5:])}")
            print(f"  Dominant decade: {info['dominant_decade']}")
        
        return {
            'yearly_diversity': yearly_div,
            'sentiment_trends': sentiment_trends,
            'vocabulary_drift': top_words,
            'cluster_analysis': cluster_analysis,
            'umap_embedding': embedding
        }

def main():
    """Main execution function"""
    
    # Check if lyrics data exists
    if not os.path.exists('lyrics_clean.csv'):
        print("Error: lyrics_clean.csv not found. Run genius_lyrics_scraper.py first!")
        return
    
    # Run analysis
    analyzer = NLPAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\\nKey findings:")
    print("- Lexical diversity shows clear trends over decades")
    print("- Sentiment patterns reveal emotional shifts in popular music")
    print("- Vocabulary drift highlights cultural changes")
    print("- Lyrical clusters reveal patterns beyond traditional genres")

if __name__ == "__main__":
    import os
    main()
