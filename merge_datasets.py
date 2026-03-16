"""
Merge Audio and Lyrics Datasets
Combines Spotify audio features and Genius lyrics for combined analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import os

class DatasetMerger:
    def __init__(self, audio_file='audio_clean.csv', lyrics_file='lyrics_clean.csv'):
        """Initialize with audio and lyrics datasets"""
        self.audio_df = pd.read_csv(audio_file)
        self.lyrics_df = pd.read_csv(lyrics_file)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare datasets for merging"""
        print(f"Audio dataset: {len(self.audio_df)} songs")
        print(f"Lyrics dataset: {len(self.lyrics_df)} songs")
        
        # Ensure consistent column names and types
        for df in [self.audio_df, self.lyrics_df]:
            df['title'] = df['title'].astype(str)
            df['artist'] = df['artist'].astype(str)
            df['year'] = df['year'].astype(int)
    
    def merge_datasets(self):
        """Merge audio and lyrics datasets"""
        # Perform inner merge on title, artist, and year
        merged_df = self.audio_df.merge(
            self.lyrics_df,
            on=['title', 'artist', 'year'],
            how='inner',
            suffixes=('_audio', '_lyrics')
        )
        
        print(f"Merged dataset: {len(merged_df)} songs")
        print(f"Merge success rate: {len(merged_df) / max(len(self.audio_df), len(self.lyrics_df)) * 100:.1f}%")
        
        # Clean up merged dataframe
        merged_df = self.clean_merged_data(merged_df)
        
        return merged_df
    
    def clean_merged_data(self, df):
        """Clean and prepare merged dataset"""
        # Add decade column
        df['decade'] = (df['year'] // 10) * 10
        df['decade'] = df['decade'].astype(str) + 's'
        
        # Add top10 binary column (using chart_position from audio data)
        df['top10'] = (df['chart_position'] <= 10).astype(int)
        
        # Select and rename key columns
        key_columns = {
            'title': 'title',
            'artist': 'artist', 
            'year': 'year',
            'decade': 'decade',
            'chart_position': 'chart_position',
            'weeks_on_chart': 'weeks_on_chart',
            'top10': 'top10',
            # Audio features
            'danceability': 'danceability',
            'energy': 'energy',
            'valence': 'valence',
            'tempo': 'tempo',
            'acousticness': 'acousticness',
            'instrumentalness': 'instrumentalness',
            'speechiness': 'speechiness',
            'loudness': 'loudness',
            # Lyrics features
            'word_count': 'word_count',
            'unique_words': 'unique_words',
            'lexical_diversity': 'lexical_diversity',
            'avg_word_length': 'avg_word_length',
            'vader_compound': 'vader_compound',
            'vader_positive': 'vader_positive',
            'vader_negative': 'vader_negative',
            'vader_neutral': 'vader_neutral',
            'textblob_polarity': 'textblob_polarity',
            'textblob_subjectivity': 'textblob_subjectivity'
        }
        
        # Keep only key columns
        df_clean = df[[col for col in key_columns.keys() if col in df.columns]].copy()
        df_clean.columns = [key_columns[col] for col in df_clean.columns]
        
        # Remove any rows with missing values in key features
        key_features = ['danceability', 'energy', 'valence', 'lexical_diversity', 'vader_compound']
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=key_features)
        final_count = len(df_clean)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing key features")
        
        print(f"Final clean dataset: {len(df_clean)} songs")
        print(f"Years covered: {df_clean['year'].min()} - {df_clean['year'].max()}")
        print(f"Decades: {', '.join(sorted(df_clean['decade'].unique()))}")
        
        return df_clean
    
    def create_ml_features(self, df):
        """Create features for machine learning models"""
        # Normalize continuous features
        feature_columns = {
            'audio_features': ['danceability', 'energy', 'valence', 'tempo', 
                              'acousticness', 'instrumentalness', 'speechiness', 'loudness'],
            'lyrics_features': ['lexical_diversity', 'word_count', 'avg_word_length',
                               'vader_compound', 'vader_positive', 'vader_negative',
                               'vader_neutral', 'textblob_polarity', 'textblob_subjectivity']
        }
        
        # Create feature scalers
        scaler_audio = StandardScaler()
        scaler_lyrics = StandardScaler()
        
        # Only use lyrics features that are actually present in the dataframe
        feature_columns['lyrics_features'] = [f for f in feature_columns['lyrics_features'] if f in df.columns]

        # Scale features
        audio_features_scaled = scaler_audio.fit_transform(df[feature_columns['audio_features']])
        lyrics_features_scaled = scaler_lyrics.fit_transform(df[feature_columns['lyrics_features']])
        
        # Create feature DataFrames
        audio_features_df = pd.DataFrame(
            audio_features_scaled,
            columns=[f"audio_{col}" for col in feature_columns['audio_features']],
            index=df.index
        )
        
        lyrics_features_df = pd.DataFrame(
            lyrics_features_scaled,
            columns=[f"lyrics_{col}" for col in feature_columns['lyrics_features']],
            index=df.index
        )

        
        # Combine all features
        ml_features = pd.concat([audio_features_df, lyrics_features_df], axis=1)
        
        # Add target variables
        ml_features['top10'] = df['top10']
        ml_features['chart_position'] = df['chart_position']
        ml_features['weeks_on_chart'] = df['weeks_on_chart']
        ml_features['decade'] = df['decade']
        
        return ml_features, scaler_audio, scaler_lyrics
    
    def analyze_feature_distributions(self, df):
        """Analyze distributions of key features by decade and success"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Key features to analyze
        key_features = ['danceability', 'energy', 'valence', 'lexical_diversity', 'vader_compound', 'tempo']
        
        for i, feature in enumerate(key_features):
            # Box plot by decade
            sns.boxplot(data=df, x='decade', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()} by Decade')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('merged_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_merged_data(self, df, filename='merged.csv'):
        """Save merged dataset"""
        df.to_csv(filename, index=False)
        print(f"Saved merged dataset to {filename}")
        
        # Also save a summary
        summary = {
            'total_songs': len(df),
            'years_range': f"{df['year'].min()}-{df['year'].max()}",
            'decades': list(sorted(df['decade'].unique())),
            'top10_songs': df['top10'].sum(),
            'top10_percentage': f"{df['top10'].mean() * 100:.1f}%",
            'features': list(df.columns)
        }
        
        with open('dataset_summary.txt', 'w') as f:
            f.write("Merged Dataset Summary\\n")
            f.write("=" * 30 + "\\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\\n")
        
        print("Dataset summary saved to dataset_summary.txt")

def main():
    """Main execution function"""
    
    # Check if required files exist
    required_files = ['audio_clean.csv', 'lyrics_clean.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please run spotify_audio_features.py and genius_lyrics_scraper.py first!")
        return
    
    # Initialize merger
    merger = DatasetMerger()
    
    # Merge datasets
    print("\\nMerging audio and lyrics datasets...")
    merged_df = merger.merge_datasets()
    
    # Create ML features
    print("\\nCreating machine learning features...")
    ml_features, scaler_audio, scaler_lyrics = merger.create_ml_features(merged_df)
    
    # Analyze feature distributions
    print("\\nAnalyzing feature distributions...")
    merger.analyze_feature_distributions(merged_df)
    
    # Save results
    merger.save_merged_data(merged_df)
    ml_features.to_csv('ml_features.csv', index=False)
    print("Saved ML features to ml_features.csv")
    
    print("\\nMerge complete! Ready for combined analysis and modeling.")
    
    return merged_df, ml_features

if __name__ == "__main__":
    main()
