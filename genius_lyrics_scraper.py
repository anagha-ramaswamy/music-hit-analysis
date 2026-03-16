"""
Genius Lyrics Scraper
Collects lyrics for songs using Genius API with validation and cleaning
"""

import pandas as pd
import lyricsgenius
import time
import os
import re
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
import requests

class GeniusLyricsCollector:
    def __init__(self, access_token):
        """Initialize Genius API client"""
        self.genius = lyricsgenius.Genius(access_token)
        self.genius.verbose = False  # Turn off status messages
        self.genius.remove_section_headers = True  # Remove [Verse], [Chorus] etc.
        self.genius.skip_non_songs = True  # Skip non-song results
        self.genius.excluded_terms = ["(Remix)", "(Live)", "(Acoustic)", "(Demo)"]  # Skip these
        
        self.lyrics_dir = "lyrics"
        self.cache_file = "lyrics_clean.csv"
        self.failures_file = "lyrics_failures.csv"
        
        # Create lyrics directory
        os.makedirs(self.lyrics_dir, exist_ok=True)
    
    def clean_lyrics(self, lyrics):
        """Clean lyrics text"""
        if not lyrics:
            return ""
        
        # Remove section headers that might remain
        lyrics = re.sub(r'\[.*?\]', '', lyrics)

        # Remove line breaks and extra spaces
        lyrics = re.sub(r'\n+', ' ', lyrics)
        lyrics = re.sub(r'\s+', ' ', lyrics)

        # Remove non-lyric content (parenthetical notes, ad-libs in parentheses)
        lyrics = re.sub(r'\([^)]*\)', '', lyrics)
        
        # Remove special characters but keep basic punctuation
        lyrics = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', lyrics)
        
        # Convert to lowercase
        lyrics = lyrics.lower().strip()
        
        return lyrics
    
    def validate_lyrics(self, lyrics, expected_artist, expected_title):
        """Validate that lyrics belong to the correct song"""
        if not lyrics or len(lyrics) < 50:  # Too short to be real lyrics
            return False, "Lyrics too short"
        
        # Check for definitive non-lyric content (metadata phrases, not individual words)
        non_lyric_indicators = [
            "copyright", "produced by", "written by", "composer", "arranged by",
            "all rights reserved", "unauthorized reproduction"
        ]

        for indicator in non_lyric_indicators:
            if indicator in lyrics.lower():
                return False, f"Contains non-lyric content: {indicator}"
        
        return True, "Valid"
    
    def get_lyrics_fallback(self, title, artist):
        """Fallback method using direct web scraping"""
        try:
            # Search for the song
            search_url = f"https://api.genius.com/search?q={title} {artist}"
            headers = {'Authorization': f'Bearer {self.genius._access_token}'}
            
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            
            results = response.json()['response']['hits']
            
            if not results:
                return None, "No search results"
            
            # Get the first result
            song_url = results[0]['result']['url']
            
            # Scrape the page
            page = requests.get(song_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            
            # Find lyrics div
            lyrics_div = soup.find('div', {'data-lyrics-container': 'true'})
            if lyrics_div:
                lyrics = lyrics_div.get_text()
                return self.clean_lyrics(lyrics), "Success"
            
            return None, "Could not find lyrics on page"
            
        except Exception as e:
            return None, f"Scraping error: {str(e)}"
    
    def get_song_lyrics(self, title, artist, max_retries=2):
        """Get lyrics for a song with validation and fallback"""
        
        for attempt in range(max_retries + 1):
            try:
                # Try lyricsgenius first
                song = self.genius.search_song(title, artist)
                
                if song:
                    # Validate the result
                    is_valid, validation_msg = self.validate_lyrics(
                        song.lyrics, artist, title
                    )
                    
                    if is_valid:
                        cleaned_lyrics = self.clean_lyrics(song.lyrics)
                        return cleaned_lyrics, "Success"
                    else:
                        print(f"Validation failed for {title} by {artist}: {validation_msg}")
                
                # If primary method fails, try fallback
                if attempt == max_retries:
                    lyrics, msg = self.get_lyrics_fallback(title, artist)
                    if lyrics:
                        return lyrics, "Fallback success"
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return None, f"Error: {str(e)}"
        
        return None, "All methods failed"
    
    def calculate_lyrics_features(self, lyrics):
        """Calculate linguistic features from lyrics"""
        if not lyrics:
            return {
                'word_count': 0,
                'unique_words': 0,
                'lexical_diversity': 0,
                'avg_word_length': 0
            }
        
        words = lyrics.split()
        unique_words = set(words)
        
        return {
            'word_count': len(words),
            'unique_words': len(unique_words),
            'lexical_diversity': len(unique_words) / len(words) if words else 0,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def process_songs(self, songs_df):
        """Process all songs and collect lyrics"""
        
        # Load existing cache if exists
        if os.path.exists(self.cache_file):
            cache_df = pd.read_csv(self.cache_file)
            processed_titles = set(zip(cache_df['title'], cache_df['artist']))
            print(f"Loaded {len(cache_df)} cached lyrics")
        else:
            cache_df = pd.DataFrame()
            processed_titles = set()
        
        # Filter songs not yet processed
        remaining_songs = songs_df[
            ~songs_df.set_index(['title', 'artist']).index.isin(processed_titles)
        ].copy()
        
        print(f"Processing {len(remaining_songs)} remaining songs...")
        
        results = []
        failures = []
        failures_df = pd.DataFrame()
        
        for idx, row in tqdm(remaining_songs.iterrows(), total=len(remaining_songs), desc="Collecting lyrics"):
            title = row['title']
            artist = row['artist']
            
            # Get lyrics
            lyrics, status = self.get_song_lyrics(title, artist)
            
            if lyrics:
                # Calculate features
                features = self.calculate_lyrics_features(lyrics)
                
                result = {
                    'title': title,
                    'artist': artist,
                    'year': row['year'],
                    'lyrics_raw': lyrics,
                    'lyrics_clean': lyrics,
                    **features
                }
                results.append(result)
                
                # Save individual lyrics file (sanitize slashes in names)
                safe_title = re.sub(r'[/\\]', '-', title)
                safe_artist = re.sub(r'[/\\]', '-', artist)
                filename = f"{self.lyrics_dir}/{safe_title}_{safe_artist}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(lyrics)
            else:
                failures.append({
                    'title': title,
                    'artist': artist,
                    'year': row['year'],
                    'error': status
                })
            
            # Rate limiting - Genius API is strict
            time.sleep(0.5)

            # Checkpoint every 10 songs
            if (len(results) + len(failures)) % 10 == 0:
                if results:
                    new_results_df = pd.DataFrame(results)
                    cache_df = pd.concat([cache_df, new_results_df], ignore_index=True)
                    cache_df.to_csv(self.cache_file, index=False)
                    results = []
                if failures:
                    existing = pd.read_csv(self.failures_file) if os.path.exists(self.failures_file) else pd.DataFrame()
                    pd.concat([existing, pd.DataFrame(failures)], ignore_index=True).to_csv(self.failures_file, index=False)
                    failures = []
                print(f"Checkpoint: {len(cache_df)} lyrics saved")

        # Save any remaining
        if results:
            new_results_df = pd.DataFrame(results)
            cache_df = pd.concat([cache_df, new_results_df], ignore_index=True)
            cache_df.to_csv(self.cache_file, index=False)
            print(f"Saved {len(results)} new lyrics")

        if failures:
            existing = pd.read_csv(self.failures_file) if os.path.exists(self.failures_file) else pd.DataFrame()
            all_failures = pd.concat([existing, pd.DataFrame(failures)], ignore_index=True)
            all_failures.to_csv(self.failures_file, index=False)
            print(f"Logged {len(all_failures)} total failures")
        
        return cache_df, failures_df

def main():
    """Main execution function"""
    
    # Load songs data
    if not os.path.exists('songs.csv'):
        print("Error: songs.csv not found. Run Phase_0_Billboard_Data.ipynb first!")
        return
    
    songs_df = pd.read_csv('songs.csv')
    print(f"Loaded {len(songs_df)} songs from songs.csv")
    
    # Get Genius API token (you'll need to set this)
    # Register at: https://genius.com/api-clients
    ACCESS_TOKEN = os.getenv('GENIUS_ACCESS_TOKEN', 'your_access_token_here')
    
    if ACCESS_TOKEN == 'your_access_token_here':
        print("Please set GENIUS_ACCESS_TOKEN environment variable")
        print("Or modify the ACCESS_TOKEN variable in this script")
        return
    
    # Initialize collector
    collector = GeniusLyricsCollector(ACCESS_TOKEN)
    
    # Process songs
    lyrics_df, failures_df = collector.process_songs(songs_df)
    
    print(f"\\nFinal results:")
    print(f"Successfully processed: {len(lyrics_df)} songs")
    print(f"Failed: {len(failures_df)} songs")
    print(f"Success rate: {len(lyrics_df) / len(songs_df) * 100:.1f}%")
    
    # Display sample statistics
    if len(lyrics_df) > 0:
        print(f"\\nLyrics statistics:")
        print(f"Average word count: {lyrics_df['word_count'].mean():.1f}")
        print(f"Average lexical diversity: {lyrics_df['lexical_diversity'].mean():.3f}")

if __name__ == "__main__":
    main()
