"""
Audio Feature Analysis
Analyzes Spotify audio features for trends and patterns over decades
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joypy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AudioAnalyzer:
    def __init__(self, audio_file='audio_clean.csv'):
        """Initialize with audio features data"""
        self.df = pd.read_csv(audio_file)
        self.audio_features = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'speechiness', 'loudness'
        ]
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for analysis"""
        # Normalize tempo to 0-1 range
        self.df['tempo_normalized'] = (self.df['tempo'] - self.df['tempo'].min()) / \
                                      (self.df['tempo'].max() - self.df['tempo'].min())
        
        # Normalize loudness to 0-1 range (loudness is negative)
        self.df['loudness_normalized'] = (self.df['loudness'] - self.df['loudness'].min()) / \
                                        (self.df['loudness'].max() - self.df['loudness'].min())
        
        # Update features list for normalized versions
        self.normalized_features = [
            'danceability', 'energy', 'valence', 'tempo_normalized',
            'acousticness', 'instrumentalness', 'speechiness', 'loudness_normalized'
        ]
        
        print(f"Loaded {len(self.df)} songs with audio features")
        print(f"Decades covered: {', '.join(sorted(self.df['decade'].unique()))}")
    
    def plot_feature_distributions(self):
        """Create KDE plots showing feature distributions by decade"""
        key_features = ['energy', 'danceability', 'valence', 'acousticness']
        decades = sorted(self.df['decade'].unique())
        colors = sns.color_palette('husl', len(decades))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, feature in enumerate(key_features):
            for j, decade in enumerate(decades):
                data = self.df[self.df['decade'] == decade][feature].dropna()
                sns.kdeplot(data, ax=axes[i], label=decade, color=colors[j],
                            fill=True, alpha=0.3, linewidth=1.5)
            axes[i].set_title(f'{feature.title()} Distribution by Decade',
                              fontsize=14, fontweight='bold')
            axes[i].set_xlabel(feature.title(), fontsize=12)
            axes[i].legend(fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig('audio_ridgeline_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_trends(self):
        """Plot mean audio features over time with confidence bands"""
        # Calculate yearly statistics
        yearly_stats = self.df.groupby('year')[self.audio_features].agg(['mean', 'std'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot trends for key features
        key_features = ['energy', 'danceability', 'valence', 'acousticness']
        
        for i, feature in enumerate(key_features):
            years = yearly_stats.index
            means = yearly_stats[feature]['mean']
            stds = yearly_stats[feature]['std']
            
            axes[i].plot(years, means, linewidth=3, color='royalblue')
            axes[i].fill_between(years, means - stds, means + stds, alpha=0.3, color='royalblue')
            
            axes[i].set_title(f'{feature.title()} Trend Over Time', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Year', fontsize=12)
            axes[i].set_ylabel(feature.title(), fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('audio_feature_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap of audio features vs chart performance"""
        # Calculate correlations
        corr_features = self.audio_features + ['chart_position', 'weeks_on_chart']
        correlation_matrix = self.df[corr_features].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Audio Features Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('audio_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_pca_analysis(self):
        """Perform PCA on audio features and visualize by decade"""
        # Prepare data
        X = self.df[self.normalized_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'decade': self.df['decade'],
            'title': self.df['title'],
            'artist': self.df['artist']
        })
        
        # Plot PCA results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Scatter plot by decade
        decades = sorted(pca_df['decade'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(decades)))
        
        for i, decade in enumerate(decades):
            data = pca_df[pca_df['decade'] == decade]
            ax1.scatter(data['PC1'], data['PC2'], 
                       c=[colors[i]], label=decade, alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax1.set_title('Audio Features PCA by Decade', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Convex hulls to show cluster tightening
        from scipy.spatial import ConvexHull
        for i, decade in enumerate(decades):
            data = pca_df[pca_df['decade'] == decade][['PC1', 'PC2']].values
            if len(data) >= 3:  # Need at least 3 points for convex hull
                hull = ConvexHull(data)
                for simplex in hull.simplices:
                    ax2.plot(data[simplex, 0], data[simplex, 1], 
                            color=colors[i], alpha=0.7, linewidth=2)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax2.set_title('Audio Feature Clusters by Decade (Convex Hulls)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('audio_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate cluster areas to quantify convergence
        cluster_areas = {}
        for decade in decades:
            data = pca_df[pca_df['decade'] == decade][['PC1', 'PC2']].values
            if len(data) >= 3:
                hull = ConvexHull(data)
                cluster_areas[decade] = hull.volume
        
        print("Cluster Areas (smaller = more converged):")
        for decade in sorted(cluster_areas.keys()):
            print(f"{decade}: {cluster_areas[decade]:.3f}")
        
        return pca_df, cluster_areas
    
    def analyze_feature_importance(self):
        """Analyze which features correlate with chart success"""
        # Create binary success metrics
        self.df['top10'] = (self.df['chart_position'] <= 10).astype(int)
        self.df['top20'] = (self.df['chart_position'] <= 20).astype(int)
        
        # Calculate feature correlations with success
        success_correlations = {}
        
        for feature in self.audio_features:
            corr_top10 = self.df[feature].corr(self.df['top10'])
            corr_weeks = self.df[feature].corr(self.df['weeks_on_chart'])
            
            success_correlations[feature] = {
                'top10_correlation': corr_top10,
                'weeks_correlation': corr_weeks
            }
        
        # Plot feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top 10 correlations
        features = list(success_correlations.keys())
        top10_corrs = [success_correlations[f]['top10_correlation'] for f in features]
        
        colors = ['red' if x < 0 else 'green' for x in top10_corrs]
        bars1 = ax1.barh(features, top10_corrs, color=colors, alpha=0.7)
        ax1.set_xlabel('Correlation with Top 10 Hit', fontsize=12)
        ax1.set_title('Audio Features vs Top 10 Success', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, corr) in enumerate(zip(bars1, top10_corrs)):
            ax1.text(corr + 0.01 if corr >= 0 else corr - 0.01, i, 
                    f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center')
        
        # Feature variance by decade (higher variance = more diverse sounds)
        decades = sorted(self.df['decade'].unique())
        decade_vars = self.df.groupby('decade')[self.audio_features].std().mean(axis=1)
        bars2 = ax2.bar(decades, decade_vars.values, alpha=0.7, color='steelblue')
        ax2.set_xlabel('Decade', fontsize=12)
        ax2.set_ylabel('Mean Feature Std Dev', fontsize=12)
        ax2.set_title('Audio Diversity by Decade\n(Lower = More Homogeneous)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        for bar, val in zip(bars2, decade_vars.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('audio_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return success_correlations
    
    def run_full_analysis(self):
        """Run complete audio analysis pipeline"""
        print("Starting Audio Feature Analysis...")
        print("=" * 50)
        
        print("\\n1. Creating feature distribution plots...")
        self.plot_feature_distributions()
        
        print("\\n2. Analyzing feature trends over time...")
        self.plot_feature_trends()
        
        print("\\n3. Creating correlation heatmap...")
        self.plot_correlation_heatmap()
        
        print("\\n4. Performing PCA analysis...")
        pca_df, cluster_areas = self.perform_pca_analysis()
        
        print("\\n5. Analyzing feature importance...")
        importance = self.analyze_feature_importance()
        
        print("\\nAnalysis complete! Check the generated PNG files.")
        
        return {
            'pca_results': pca_df,
            'cluster_areas': cluster_areas,
            'feature_importance': importance
        }

def main():
    """Main execution function"""
    
    # Check if audio data exists
    if not os.path.exists('audio_clean.csv'):
        print("Error: audio_clean.csv not found. Run spotify_audio_features.py first!")
        return
    
    # Run analysis
    analyzer = AudioAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\\nKey findings:")
    print("- Audio feature distributions show clear patterns by decade")
    print("- PCA reveals clustering patterns that may indicate convergence")
    print("- Certain audio features correlate more strongly with chart success")

if __name__ == "__main__":
    import os
    main()
