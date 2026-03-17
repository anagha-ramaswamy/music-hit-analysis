"""
Final Combined Analysis
Machine learning models and final visualizations for "The Anatomy of a Hit"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinalAnalyzer:
    def __init__(self, merged_file='merged.csv'):
        """Initialize with merged dataset"""
        self.df = pd.read_csv(merged_file)
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analysis"""
        if 'decade' not in self.df.columns:
            self.df['decade'] = (self.df['year'] // 10 * 10).astype(str) + 's'

        print(f"Loaded merged dataset: {len(self.df)} songs")
        print(f"Years: {self.df['year'].min()} - {self.df['year'].max()}")
        print(f"Decades: {', '.join(sorted(self.df['decade'].unique()))}")

        # Define feature groups
        self.audio_features = ['danceability', 'energy', 'valence', 'tempo',
                              'acousticness', 'instrumentalness', 'speechiness', 'loudness']

        self.lyrics_features = ['lexical_diversity', 'word_count', 'avg_word_length',
                               'compressibility',
                               'vader_compound', 'vader_positive', 'vader_negative',
                               'vader_neutral', 'textblob_polarity', 'textblob_subjectivity']

        self.lyrics_features = [f for f in self.lyrics_features if f in self.df.columns]
        self.all_features = self.audio_features + self.lyrics_features

    def train_predictive_models(self):
        """Train decade prediction models — tests homogenization hypothesis directly.
        High accuracy = decades sound distinct. Low accuracy = convergence."""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X = self.df[self.all_features].fillna(self.df[self.all_features].mean())
        y = le.fit_transform(self.df['decade'])
        self.decade_labels = le.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='accuracy')
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, scaler.transform(X), y, cv=StratifiedKFold(5), scoring='accuracy')

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Macro F1: {f1:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            print(f"\n  Confusion matrix (rows=true, cols=predicted):")
            header = '        ' + ''.join(f'{d:>8}' for d in le.classes_)
            print(header)
            for i, decade in enumerate(le.classes_):
                row = f'  {decade}  ' + ''.join(f'{cm_norm[i,j]:>8.2f}' for j in range(len(le.classes_)))
                print(row)
            print(f"\n  Per-decade recall:")
            for decade, acc in zip(le.classes_, per_class_acc):
                bar = '█' * int(acc * 20)
                print(f"    {decade}: {acc:.2f} {bar}")

        return results, X_train, X_test, y_train, y_test, scaler

    def plot_shap_analysis(self, results, X_train, X_test):
        """SHAP values for interpretable feature importance → shap_analysis.png
        Audio vs lyrics split cited in poster text."""
        rf_model = results['Random Forest']['model']

        print("Computing SHAP values (this may take a minute)...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)

        # Handle different SHAP output formats across versions
        # For multiclass (6 decades), average abs SHAP across all classes
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            sv = shap_values[0]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values
            mean_shap = np.abs(sv).mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        shap_df = pd.DataFrame({
            'feature': self.all_features,
            'mean_shap': mean_shap,
            'type': ['Audio'] * len(self.audio_features) + ['Lyrics'] * len(self.lyrics_features)
        }).sort_values('mean_shap', ascending=True)

        colors = ['lightcoral' if t == 'Audio' else 'lightblue' for t in shap_df['type']]
        axes[0].barh(shap_df['feature'], shap_df['mean_shap'], color=colors, alpha=0.85)
        axes[0].set_xlabel('Mean |SHAP value|', fontsize=12)
        axes[0].set_title('SHAP Feature Importance\n(Average impact on decade classification)',
                          fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color='lightcoral', label='Audio'),
                                 Patch(color='lightblue', label='Lyrics')], loc='lower right')

        # SHAP scatter for top feature
        top_feat_idx = int(np.argmax(mean_shap))
        top_feat_name = self.all_features[top_feat_idx]
        axes[1].scatter(X_test.iloc[:, top_feat_idx], sv[:, top_feat_idx],
                        alpha=0.4, c=sv[:, top_feat_idx], cmap='RdYlGn', s=30)
        axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1].set_xlabel(top_feat_name.replace('_', ' ').title(), fontsize=12)
        axes[1].set_ylabel('SHAP Value', fontsize=12)
        axes[1].set_title(f'SHAP Dependence: {top_feat_name.replace("_", " ").title()}',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Report audio vs lyrics split
        audio_shap = shap_df[shap_df['type'] == 'Audio']['mean_shap'].sum()
        lyrics_shap = shap_df[shap_df['type'] == 'Lyrics']['mean_shap'].sum()
        total = audio_shap + lyrics_shap
        print(f"\nSHAP split: Audio {audio_shap/total:.1%} vs Lyrics {lyrics_shap/total:.1%}")

        return shap_df

    def analyze_temporal_patterns(self):
        """Lexical diversity + sentiment trends over time → temporal_patterns_analysis.png"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Lexical diversity over time with quadratic trend
        yearly_lex = self.df.groupby('year')['lexical_diversity'].mean()
        axes[0].plot(yearly_lex.index, yearly_lex.values, linewidth=3, color='purple')
        z = np.polyfit(yearly_lex.index, yearly_lex.values, 2)
        p = np.poly1d(z)
        residuals = yearly_lex.values - p(yearly_lex.index)
        r2 = 1 - np.sum(residuals**2) / np.sum((yearly_lex.values - yearly_lex.values.mean())**2)
        min_year = int(-z[1] / (2 * z[0]))
        axes[0].plot(yearly_lex.index, p(yearly_lex.index), 'r--', alpha=0.8,
                     label=f'Quadratic trend (R²={r2:.2f}, min ≈ {min_year})')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Lexical Diversity', fontsize=12)
        axes[0].set_title('Lexical Diversity Over Time', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Sentiment over time with linear trend
        sentiment_col = 'textblob_polarity' if 'textblob_polarity' in self.df.columns else 'vader_compound'
        yearly_sentiment = self.df.groupby('year')[sentiment_col].mean()
        axes[1].plot(yearly_sentiment.index, yearly_sentiment.values, linewidth=3, color='green')
        z2 = np.polyfit(yearly_sentiment.index, yearly_sentiment.values, 1)
        p2 = np.poly1d(z2)
        from scipy import stats as scipy_stats
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(
            yearly_sentiment.index, yearly_sentiment.values)
        axes[1].plot(yearly_sentiment.index, p2(yearly_sentiment.index), 'r--', alpha=0.8,
                     label=f'Trend (slope={slope:.4f}/yr, p={p_val:.3f})')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Sentiment (TextBlob polarity)', fontsize=12)
        axes[1].set_title('Sentiment Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_decade_similarity_heatmap(self):
        """Cosine similarity between decade centroids in audio feature space → decade_similarity_heatmap.png"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(
            self.df[self.audio_features].fillna(self.df[self.audio_features].mean())
        )

        decades = sorted(self.df['decade'].unique())
        centroids = np.array([
            X_scaled[self.df['decade'].values == d].mean(axis=0) for d in decades
        ])

        sim_matrix = cosine_similarity(centroids)
        sim_df = pd.DataFrame(sim_matrix, index=decades, columns=decades)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Full similarity heatmap
        sns.heatmap(sim_df, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=sim_matrix.min() - 0.01, vmax=1.0, ax=axes[0],
                    linewidths=0.5, cbar_kws={'label': 'Cosine Similarity'})
        axes[0].set_title('Decade-to-Decade Sonic Similarity\n(1.000 = identical average sound)',
                          fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Decade')
        axes[0].set_ylabel('Decade')

        # Similarity to 1970s baseline
        baseline = sim_df.loc['1970s'].drop('1970s')
        colors = sns.color_palette('husl', len(baseline))
        bars = axes[1].bar(baseline.index, baseline.values, color=colors, alpha=0.8)
        y_min = baseline.values.min() - 0.005
        axes[1].set_ylim(y_min, 1.002)
        axes[1].set_xlabel('Decade', fontsize=12)
        axes[1].set_ylabel('Cosine Similarity to 1970s', fontsize=12)
        axes[1].set_title('How Far Has Each Decade Drifted\nFrom the 1970s Sound?',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        for bar, val in zip(bars, baseline.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                         f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('decade_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nDecade Cosine Similarity Matrix:")
        print(sim_df.round(4).to_string())
        return sim_df

    def plot_decade_dendrogram(self):
        """Hierarchical clustering of decade audio centroids (Ward's method) → decade_dendrogram.png"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(
            self.df[self.audio_features].fillna(self.df[self.audio_features].mean())
        )

        decades = sorted(self.df['decade'].unique())
        centroids = np.array([
            X_scaled[self.df['decade'].values == d].mean(axis=0) for d in decades
        ])

        Z = linkage(centroids, method='ward')

        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(
            Z,
            labels=decades,
            ax=ax,
            color_threshold=0.7 * max(Z[:, 2]),
            above_threshold_color='#888888',
            leaf_font_size=14,
        )
        ax.set_title('Hierarchical Clustering of Decade Audio Profiles\n(Ward linkage on 8 standardized audio features)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Decade', fontsize=12)
        ax.set_ylabel('Ward Distance', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('decade_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nDecade Dendrogram linkage matrix (Ward):")
        print(pd.DataFrame(Z, columns=['idx1', 'idx2', 'distance', 'count']).round(4).to_string())
        return Z

    def plot_kmeans_cluster_heatmap(self):
        """K-Means (k=6) on combined audio+lyric features; ARI cited in poster → kmeans_cluster_heatmap.png"""
        feature_cols = [f for f in self.audio_features + ['lexical_diversity', 'compressibility', 'textblob_polarity']
                        if f in self.df.columns]
        sub = self.df[feature_cols + ['decade']].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[feature_cols])

        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        sub = sub.copy()
        sub['cluster'] = kmeans.fit_predict(X_scaled)

        # Cross-tab: rows = clusters, cols = decades, values = % of cluster in each decade
        crosstab = pd.crosstab(sub['cluster'], sub['decade'])
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        crosstab_pct.index = [f'Cluster {i}' for i in crosstab_pct.index]

        # Adjusted Rand Index
        decade_codes = sub['decade'].astype('category').cat.codes
        ari = adjusted_rand_score(decade_codes, sub['cluster'])

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(crosstab_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, cbar_kws={'label': '% of cluster songs'})
        ax.set_title(
            f'K-Means Cluster Composition by Decade (k=6, combined audio + lyric features)\n'
            f'Adjusted Rand Index = {ari:.3f}  (0 = random, 1 = perfect decade alignment)',
            fontsize=13, fontweight='bold')
        ax.set_xlabel('Decade', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        plt.tight_layout()
        plt.savefig('kmeans_cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nAdjusted Rand Index (clusters vs decades): {ari:.4f}")
        print("\nCluster composition (% per decade):")
        print(crosstab_pct.round(1).to_string())

        return crosstab_pct, ari

    def run_complete_analysis(self):
        """Run the complete final analysis pipeline"""
        print("Starting Final Combined Analysis...")
        print("=" * 50)

        print("\n1. Training predictive models...")
        results, X_train, X_test, y_train, y_test, scaler = self.train_predictive_models()

        print("\n2. SHAP analysis...")
        shap_df = self.plot_shap_analysis(results, X_train, X_test)

        print("\n3. Temporal patterns (lexical diversity + sentiment)...")
        self.analyze_temporal_patterns()

        print("\n4. Decade sonic similarity heatmap...")
        self.plot_decade_similarity_heatmap()

        print("\n5. Hierarchical clustering dendrogram...")
        self.plot_decade_dendrogram()

        print("\n6. K-Means cluster heatmap (combined features)...")
        self.plot_kmeans_cluster_heatmap()

        print("\nAnalysis complete! Check the generated PNG files.")

        best_model = max(results, key=lambda m: results[m]['accuracy'])
        print(f"\nBest decade classifier: {best_model} with {results[best_model]['accuracy']:.1%} accuracy")

        return results, shap_df


def main():
    if not os.path.exists('merged.csv'):
        print("Error: merged.csv not found. Run merge_datasets.py first!")
        return

    analyzer = FinalAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
