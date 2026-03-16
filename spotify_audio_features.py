"""
Audio Feature Collection (YouTube + librosa)

Searches YouTube for each song, downloads 30 seconds of audio using yt-dlp,
then extracts audio features locally using librosa.

Feature mapping:
  tempo           : beat tracking BPM
  energy          : normalized RMS energy
  loudness        : mean loudness in dB
  danceability    : normalized onset strength (rhythmic regularity proxy)
  acousticness    : inverse normalized spectral centroid (lower brightness = more acoustic)
  speechiness     : zero crossing rate (higher = more speech-like)
  instrumentalness: inverse ZCR (higher = less vocal)
  valence         : major/minor key detection via Krumhansl-Schmuckler profiles
                    combined with tempo and energy (major key = higher valence)
"""

import os
import re
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


class AudioCollector:
    def __init__(self):
        self.cache_file = "audio_raw.csv"
        self.failures_file = "audio_failures.csv"
        self._check_ytdlp()

    def _check_ytdlp(self):
        """Verify yt-dlp is installed"""
        import sys
        result = subprocess.run([sys.executable, '-m', 'yt_dlp', '--version'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("yt-dlp not found. Install with: pip3 install yt-dlp")
        print(f"yt-dlp version: {result.stdout.strip()}")

    @staticmethod
    def clean_name(s):
        """Strip featured artists, parentheticals for cleaner search queries"""
        s = re.sub(r'\(feat\..*?\)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\(ft\..*?\)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'feat\..*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'ft\..*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'featuring.*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\(.*?\)', '', s)
        return s.strip()

    def download_audio(self, title, artist, duration=30):
        """Search YouTube and download first 30 seconds of audio"""
        query = f"{self.clean_name(title)} {self.clean_name(artist)} official audio"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'audio.%(ext)s')

            import sys
            cmd = [
                sys.executable, '-m', 'yt_dlp',
                f'ytsearch1:{query}',       # take first YouTube result
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '5',     # medium quality, faster download
                '--download-sections', f'*0-{duration}',  # first 30 seconds only
                '--output', out_path,
                '--quiet',
                '--no-warnings',
                '--no-playlist',
            ]

            env = os.environ.copy()
            env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + env.get('PATH', '')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)

            if result.returncode != 0:
                print(f"yt-dlp error for '{title}': {result.stderr[:200]}")
                return None

            # Find the downloaded file
            wav_files = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
            if not wav_files:
                return None

            wav_path = os.path.join(tmpdir, wav_files[0])
            return self.extract_librosa_features(wav_path)

    def extract_librosa_features(self, wav_path):
        """Extract audio features from a wav file using librosa"""
        try:
            y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=30.0)

            # Tempo (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])

            # Energy: use perceptual loudness (LUFS-like), mapped from typical range
            rms = librosa.feature.rms(y=y)[0]
            loudness_db = float(librosa.amplitude_to_db(rms).mean())
            # Typical music ranges from -40dB (quiet) to -5dB (loud)
            energy = float(np.clip((loudness_db + 40) / 35, 0, 1))

            # Loudness (dB) - raw value for analysis
            loudness = loudness_db

            # Danceability: combine rhythmic regularity + tempo stability
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # onset_strength mean typically ranges 1-8 for music
            danceability = float(np.clip(np.mean(onset_env) / 6.0, 0, 1))

            # Spectral centroid (brightness, normalized)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_norm = float(np.mean(centroid)) / (sr / 2)

            # Acousticness: inverse brightness (centroid_norm typically 0.05-0.25)
            acousticness = float(np.clip(1.0 - centroid_norm * 8, 0, 1))

            # Speechiness: use spectral flatness (tonal=instrumental, flat=speech/noise)
            # Low flatness = tonal (instrumental), high flatness = noisy/speech
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            # flatness typically 0-0.1 for music, higher for speech
            speechiness = float(np.clip(np.mean(flatness) / 0.05, 0, 1))
            instrumentalness = float(np.clip(1.0 - speechiness * 1.5, 0, 1))

            # Valence: Krumhansl-Schmuckler key profiles
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            major_scores = [np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
                            for i in range(12)]
            minor_scores = [np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
                            for i in range(12)]
            is_major = float(max(major_scores) >= max(minor_scores))
            tempo_norm = float(np.clip((tempo - 60) / 140, 0, 1))
            valence = float(0.5 * is_major + 0.3 * tempo_norm + 0.2 * energy)

            return {
                'tempo': tempo,
                'energy': energy,
                'loudness': loudness,
                'danceability': danceability,
                'acousticness': acousticness,
                'speechiness': speechiness,
                'instrumentalness': instrumentalness,
                'valence': valence,
            }

        except Exception as e:
            print(f"librosa extraction error: {e}")
            return None

    def process_songs(self, songs_df):
        """Process all songs and collect audio features"""
        if os.path.exists(self.cache_file):
            cache_df = pd.read_csv(self.cache_file)
            processed = set(zip(cache_df['title'], cache_df['artist']))
            print(f"Loaded {len(cache_df)} cached results")
        else:
            cache_df = pd.DataFrame()
            processed = set()

        remaining = songs_df[
            ~songs_df.set_index(['title', 'artist']).index.isin(processed)
        ].copy()
        print(f"Processing {len(remaining)} remaining songs...")

        results = []
        failures = []
        songs_processed = 0

        for idx, row in tqdm(remaining.iterrows(), total=len(remaining),
                             desc="Collecting audio features"):
            title, artist = row['title'], row['artist']

            features = self.download_audio(title, artist)

            if features:
                results.append({
                    'title': title,
                    'artist': artist,
                    'year': row['year'],
                    'chart_position': row['chart_position'],
                    'weeks_on_chart': row['weeks_on_chart'],
                    **features,
                })
            else:
                failures.append({
                    'title': title,
                    'artist': artist,
                    'year': row['year'],
                    'error': 'download_failed',
                })

            songs_processed += 1

            # Checkpoint every 10 songs
            if songs_processed % 10 == 0:
                if results:
                    new_df = pd.DataFrame(results)
                    cache_df = pd.concat([cache_df, new_df], ignore_index=True)
                    cache_df.to_csv(self.cache_file, index=False)
                    results = []
                if failures:
                    existing = pd.read_csv(self.failures_file) if os.path.exists(self.failures_file) else pd.DataFrame()
                    pd.concat([existing, pd.DataFrame(failures)], ignore_index=True).to_csv(self.failures_file, index=False)
                    failures = []
                print(f"Checkpoint: {len(cache_df)} saved, {songs_processed} processed")

        # Save any remaining
        if results:
            new_df = pd.DataFrame(results)
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            cache_df.to_csv(self.cache_file, index=False)
        if failures:
            existing = pd.read_csv(self.failures_file) if os.path.exists(self.failures_file) else pd.DataFrame()
            all_failures = pd.concat([existing, pd.DataFrame(failures)], ignore_index=True)
            all_failures.to_csv(self.failures_file, index=False)
            print(f"Logged {len(all_failures)} total failures")

        return cache_df, pd.DataFrame(failures)

    def clean_audio_data(self, df):
        """Clean and validate audio feature data"""
        df['decade'] = (df['year'] // 10) * 10
        df['decade'] = df['decade'].astype(str) + 's'

        print(f"Before cleaning: {len(df)} songs")
        df = df[(df['tempo'] > 40) & (df['tempo'] < 220)]

        key_features = ['danceability', 'energy', 'valence', 'tempo',
                        'acousticness', 'instrumentalness', 'speechiness', 'loudness']
        df = df.dropna(subset=key_features)

        print(f"After cleaning: {len(df)} songs")
        df.to_csv('audio_clean.csv', index=False)
        print("Saved audio_clean.csv")
        return df


def main():
    if not os.path.exists('songs.csv'):
        print("Error: songs.csv not found. Run Phase_0_Billboard_Data.ipynb first!")
        return

    songs_df = pd.read_csv('songs.csv')
    print(f"Loaded {len(songs_df)} songs from songs.csv")

    collector = AudioCollector()
    audio_df, failures_df = collector.process_songs(songs_df)
    clean_df = collector.clean_audio_data(audio_df)

    print(f"\nFinal results:")
    print(f"Successfully processed: {len(clean_df)} songs")
    print(f"Failed: {len(failures_df)} songs")
    print(f"Success rate: {len(clean_df) / len(songs_df) * 100:.1f}%")


if __name__ == "__main__":
    main()
