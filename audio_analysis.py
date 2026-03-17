"""
Audio Feature Analysis
Analyzes librosa audio features for trends and patterns over decades
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

        self.normalized_features = [
            'danceability', 'energy', 'valence', 'tempo_normalized',
            'acousticness', 'instrumentalness', 'speechiness', 'loudness_normalized'
        ]

        print(f"Loaded {len(self.df)} songs with audio features")
        print(f"Decades covered: {', '.join(sorted(self.df['decade'].unique()))}")

    def plot_feature_trends(self):
        """Plot mean audio features over time with confidence bands → audio_feature_trends.png"""
        yearly_stats = self.df.groupby('year')[self.audio_features].agg(['mean', 'std'])

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, feature in enumerate(self.audio_features):
            years = yearly_stats.index
            means = yearly_stats[feature]['mean']
            stds = yearly_stats[feature]['std']

            axes[i].plot(years, means, linewidth=3, color='royalblue')
            axes[i].fill_between(years, means - stds, means + stds, alpha=0.3, color='royalblue')

            axes[i].set_title(f'{feature.replace("_", " ").title()} Trend Over Time',
                              fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Year', fontsize=10)
            axes[i].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('audio_feature_trends.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_full_analysis(self):
        """Run complete audio analysis pipeline"""
        print("Starting Audio Feature Analysis...")
        print("=" * 50)

        print("\n1. Analyzing feature trends over time...")
        self.plot_feature_trends()

        print("\nAnalysis complete! Check audio_feature_trends.png")


def main():
    if not os.path.exists('audio_clean.csv'):
        print("Error: audio_clean.csv not found.")
        return

    analyzer = AudioAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
