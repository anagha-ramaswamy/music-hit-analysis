"""
Merge Audio and Lyrics Datasets
Combines librosa audio features and Genius lyrics for combined analysis
"""

import pandas as pd
import numpy as np
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

        # Drop columns from lyrics_df that already exist in audio_df (besides the join keys)
        # to avoid _audio/_lyrics suffixes after merge (e.g. chart_position)
        join_keys = {'title', 'artist', 'year'}
        audio_cols = set(self.audio_df.columns) - join_keys
        duplicate_cols = [c for c in self.lyrics_df.columns
                          if c in audio_cols and c not in join_keys]
        if duplicate_cols:
            self.lyrics_df = self.lyrics_df.drop(columns=duplicate_cols)
            print(f"Dropped duplicate columns from lyrics before merge: {duplicate_cols}")

        # Ensure consistent column names and types
        for df in [self.audio_df, self.lyrics_df]:
            df['title'] = df['title'].astype(str)
            df['artist'] = df['artist'].astype(str)
            df['year'] = df['year'].astype(int)

    def merge_datasets(self):
        """Merge audio and lyrics datasets on title, artist, year"""
        merged_df = self.audio_df.merge(
            self.lyrics_df,
            on=['title', 'artist', 'year'],
            how='inner',
            suffixes=('_audio', '_lyrics')
        )

        print(f"Merged dataset: {len(merged_df)} songs")
        print(f"Merge success rate: {len(merged_df) / max(len(self.audio_df), len(self.lyrics_df)) * 100:.1f}%")

        merged_df = self.clean_merged_data(merged_df)
        return merged_df

    def clean_merged_data(self, df):
        """Clean and prepare merged dataset"""
        df['decade'] = (df['year'] // 10) * 10
        df['decade'] = df['decade'].astype(str) + 's'

        df['top10'] = (df['chart_position'] <= 10).astype(int)

        # Select key columns
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
            'compressibility': 'compressibility',
            'vader_compound': 'vader_compound',
            'vader_positive': 'vader_positive',
            'vader_negative': 'vader_negative',
            'vader_neutral': 'vader_neutral',
            'textblob_polarity': 'textblob_polarity',
            'textblob_subjectivity': 'textblob_subjectivity'
        }

        df_clean = df[[col for col in key_columns.keys() if col in df.columns]].copy()
        df_clean.columns = [key_columns[col] for col in df_clean.columns]

        # Remove rows with missing values in key features
        key_features = ['danceability', 'energy', 'valence', 'lexical_diversity', 'vader_compound']
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[f for f in key_features if f in df_clean.columns])
        final_count = len(df_clean)

        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing key features")

        print(f"Final clean dataset: {len(df_clean)} songs")
        print(f"Years covered: {df_clean['year'].min()} - {df_clean['year'].max()}")
        print(f"Decades: {', '.join(sorted(df_clean['decade'].unique()))}")

        return df_clean

    def save_merged_data(self, df, filename='merged.csv'):
        """Save merged dataset"""
        df.to_csv(filename, index=False)
        print(f"Saved merged dataset to {filename}")


def main():
    required_files = ['audio_clean.csv', 'lyrics_clean.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return

    merger = DatasetMerger()

    print("\nMerging audio and lyrics datasets...")
    merged_df = merger.merge_datasets()

    merger.save_merged_data(merged_df)

    print("\nMerge complete! Ready for final_analysis.py.")
    return merged_df


if __name__ == "__main__":
    main()
