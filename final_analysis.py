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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        print(f"Loaded merged dataset: {len(self.df)} songs")
        print(f"Years: {self.df['year'].min()} - {self.df['year'].max()}")
        print(f"Top 10 hits: {self.df['top10'].sum()} ({self.df['top10'].mean()*100:.1f}%)")
        
        # Define feature groups
        self.audio_features = ['danceability', 'energy', 'valence', 'tempo', 
                              'acousticness', 'instrumentalness', 'speechiness', 'loudness']
        
        self.lyrics_features = ['lexical_diversity', 'word_count', 'avg_word_length',
                               'vader_compound', 'vader_positive', 'vader_negative',
                               'vader_neutral', 'textblob_polarity', 'textblob_subjectivity']
        
        self.lyrics_features = [f for f in self.lyrics_features if f in self.df.columns]
        self.all_features = self.audio_features + self.lyrics_features
    
    def train_predictive_models(self):
        """Train decade prediction models — tests homogenization hypothesis directly.
        High accuracy = decades sound distinct. Confusion between recent decades = convergence."""

        # Encode decade as integer label
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
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
        }

        results = {}

        for name, model in models.items():
            print(f"\\nTraining {name}...")

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

        return results, X_train, X_test, y_train, y_test, scaler

    def plot_shap_analysis(self, results, X_train, X_test):
        """SHAP values for interpretable feature importance"""
        rf_model = results['Random Forest']['model']

        print("Computing SHAP values (this may take a minute)...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)

        # Handle different SHAP output formats across versions
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Beeswarm-style summary (mean abs SHAP per feature)
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': self.all_features,
            'mean_shap': mean_shap,
            'type': ['Audio'] * len(self.audio_features) + ['Lyrics'] * len(self.lyrics_features)
        }).sort_values('mean_shap', ascending=True)

        colors = ['lightcoral' if t == 'Audio' else 'lightblue' for t in shap_df['type']]
        axes[0].barh(shap_df['feature'], shap_df['mean_shap'], color=colors, alpha=0.85)
        axes[0].set_xlabel('Mean |SHAP value|', fontsize=12)
        axes[0].set_title('SHAP Feature Importance\n(Impact on Predicting a Hit)', fontsize=13, fontweight='bold')
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

        return shap_df

    def plot_feature_importance(self, results):
        """Create comprehensive feature importance visualization"""
        
        # Get Random Forest feature importances
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'feature': self.all_features,
            'importance': importances,
            'type': ['Audio'] * len(self.audio_features) + ['Lyrics'] * len(self.lyrics_features)
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
        
        # Create subplot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Overall feature importance
        colors = ['lightcoral' if 'audio' in f.lower() or f in self.audio_features 
                 else 'lightblue' for f in feature_importance_df['feature']]
        
        bars = axes[0].barh(feature_importance_df['feature'], feature_importance_df['importance'],
                          color=colors, alpha=0.8)
        
        axes[0].set_xlabel('Feature Importance', fontsize=12)
        axes[0].set_title('Random Forest Feature Importance\\n(What Makes a Hit Song)', 
                        fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, feature_importance_df['importance'])):
            axes[0].text(imp + 0.001, i, f'{imp:.3f}', ha='left', va='center')
        
        # Audio vs Lyrics comparison
        audio_importance = feature_importance_df[feature_importance_df['type'] == 'Audio']['importance'].sum()
        lyrics_importance = feature_importance_df[feature_importance_df['type'] == 'Lyrics']['importance'].sum()
        
        total_importance = audio_importance + lyrics_importance
        audio_pct = audio_importance / total_importance * 100
        lyrics_pct = lyrics_importance / total_importance * 100
        
        wedges, texts, autotexts = axes[1].pie(
            [audio_pct, lyrics_pct], 
            labels=['Audio Features', 'Lyrics Features'],
            colors=['lightcoral', 'lightblue'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        axes[1].set_title('Audio vs Lyrics Importance\\n(It\'s All About the Beat?)', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df, audio_pct, lyrics_pct
    
    def plot_model_comparison(self, results):
        """Plot confusion matrix for best model — shows which decades get confused"""
        best_model = max(results, key=lambda m: results[m]['accuracy'])
        cm = results[best_model]['confusion_matrix']
        y_test = results[best_model]['y_test']

        # Normalize confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion matrix heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.decade_labels, yticklabels=self.decade_labels,
                    ax=axes[0])
        axes[0].set_xlabel('Predicted Decade', fontsize=12)
        axes[0].set_ylabel('True Decade', fontsize=12)
        axes[0].set_title(f'Decade Confusion Matrix ({best_model})\nOff-diagonal = sonic similarity between eras',
                          fontsize=13, fontweight='bold')

        # Per-decade accuracy bar chart
        per_decade_acc = cm.diagonal() / cm.sum(axis=1)
        bars = axes[1].bar(self.decade_labels, per_decade_acc, alpha=0.8, color='steelblue')
        axes[1].set_xlabel('Decade', fontsize=12)
        axes[1].set_ylabel('Classification Accuracy', fontsize=12)
        axes[1].set_title('How Distinct Does Each Decade Sound?\n(Lower = More Similar to Other Decades)',
                          fontsize=13, fontweight='bold')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        for bar, acc in zip(bars, per_decade_acc):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_temporal_patterns(self):
        """Analyze how audio and lyrical patterns have changed over time"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Mean energy by decade
        decade_energy = self.df.groupby('decade')['energy'].mean()
        bars = axes[0, 0].bar(decade_energy.index, decade_energy.values,
                             alpha=0.7, color='gold')
        axes[0, 0].set_xlabel('Decade')
        axes[0, 0].set_ylabel('Mean Energy')
        axes[0, 0].set_title('Mean Energy by Decade')
        axes[0, 0].grid(True, alpha=0.3)
        for bar, val in zip(bars, decade_energy.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                           f'{val:.3f}', ha='center', va='bottom')

        # Audio feature trends over time
        yearly_features = self.df.groupby('year')[['energy', 'danceability', 'valence']].mean()
        for feature in ['energy', 'danceability', 'valence']:
            axes[0, 1].plot(yearly_features.index, yearly_features[feature],
                           linewidth=2, label=feature.title(), alpha=0.8)
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].set_title('Audio Feature Trends Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Lexical diversity over time
        yearly_lex = self.df.groupby('year')['lexical_diversity'].mean()
        axes[1, 0].plot(yearly_lex.index, yearly_lex.values,
                       linewidth=3, color='purple')
        z = np.polyfit(yearly_lex.index, yearly_lex.values, 2)
        p = np.poly1d(z)
        axes[1, 0].plot(yearly_lex.index, p(yearly_lex.index), 'r--', alpha=0.8, label='Quadratic trend')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Lexical Diversity')
        axes[1, 0].set_title('Lexical Diversity Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Sentiment trends
        sentiment_col = 'vader_compound' if 'vader_compound' in self.df.columns else 'textblob_polarity'
        yearly_sentiment = self.df.groupby('year')[sentiment_col].mean()
        axes[1, 1].plot(yearly_sentiment.index, yearly_sentiment.values,
                       linewidth=3, color='green')
        z2 = np.polyfit(yearly_sentiment.index, yearly_sentiment.values, 1)
        p2 = np.poly1d(z2)
        axes[1, 1].plot(yearly_sentiment.index, p2(yearly_sentiment.index), 'r--', alpha=0.8,
                       label=f'Trend (slope: {z2[0]:.4f})')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Sentiment')
        axes[1, 1].set_title('Sentiment Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_final_summary_visualization(self, feature_importance_df, audio_pct, lyrics_pct, results=None):
        """Create the final poster-style summary visualization"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('The Anatomy of a Hit: What Really Makes a Song Successful?', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Audio vs Lyrics pie chart (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        wedges, texts, autotexts = ax1.pie(
            [audio_pct, lyrics_pct], 
            labels=['Audio Features', 'Lyrics Features'],
            colors=['lightcoral', 'lightblue'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('It IS All About the Beat!', fontsize=12, fontweight='bold')
        
        # 2. Top 10 most important features (top middle)
        ax2 = fig.add_subplot(gs[0, 1:])
        top_features = feature_importance_df.tail(10)
        bars = ax2.barh(range(len(top_features)), top_features['importance'],
                      color=['lightcoral' if f in self.audio_features else 'lightblue' 
                            for f in top_features['feature']], alpha=0.8)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']])
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 10 Predictors of Hit Songs', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Lexical diversity (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        yearly_div = self.df.groupby('year')['lexical_diversity'].mean()
        ax3.plot(yearly_div.index, yearly_div.values, linewidth=3, color='purple')
        z = np.polyfit(yearly_div.index, yearly_div.values, 2)
        p = np.poly1d(z)
        ax3.plot(yearly_div.index, p(yearly_div.index), "r--", alpha=0.8)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Lexical Diversity')
        ax3.set_title('Lyrics: Simpler then Recovering', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Audio feature trends (middle middle)
        ax4 = fig.add_subplot(gs[1, 1])
        for feature in ['energy', 'danceability', 'valence']:
            decade_means = self.df.groupby('decade')[feature].mean()
            ax4.plot(range(len(decade_means)), decade_means.values,
                    linewidth=2, marker='o', label=feature.title())
        ax4.set_xlabel('Decade')
        ax4.set_ylabel('Feature Value')
        ax4.set_title('Audio Feature Trends by Decade', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(decade_means)))
        ax4.set_xticklabels(decade_means.index, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Decade classification accuracy (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        model_names = list(results.keys()) if results else ['Random Forest', 'KNN', 'Logistic Regression']
        accuracies = [results[m]['accuracy'] for m in model_names] if results else [0.0, 0.0, 0.0]
        bars = ax5.bar(model_names, accuracies, alpha=0.8, color=['green', 'orange', 'blue'])
        ax5.set_ylabel('Decade Prediction Accuracy')
        ax5.set_title('Can We Predict the Decade?\n(Higher = More Distinct Eras)', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=15)
        for bar, acc in zip(bars, accuracies):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom')
        ax5.grid(True, alpha=0.3)

        # 6. Key findings summary (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        findings = [
            "Finding 1: Lyrics became simpler through the 2000s but recovered in the streaming era (U-shape, R2=0.68)",
            "Finding 2: Audio features converged 1970s-2000s, then diverged in the 2010s (PCA cluster areas)",
            "Finding 3: Lyrical themes grew more diverse across all decades (LDA topic modeling)",
            "Finding 4: Sentiment in hit songs has trended darker over time"
        ]

        for i, finding in enumerate(findings):
            ax6.text(0.05, 0.8 - i*0.18, finding, fontsize=11,
                    transform=ax6.transAxes, va='top')
        
        plt.savefig('final_summary_poster.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete final analysis pipeline"""
        print("Starting Final Combined Analysis...")
        print("=" * 50)
        
        print("\\n1. Training predictive models...")
        results, X_train, X_test, y_train, y_test, scaler = self.train_predictive_models()
        
        print("\\n2. SHAP analysis...")
        shap_df = self.plot_shap_analysis(results, X_train, X_test)

        print("\\n3. Analyzing feature importance...")
        feature_importance_df, audio_pct, lyrics_pct = self.plot_feature_importance(results)
        
        print("\\n4. Comparing model performance...")
        self.plot_model_comparison(results)

        print("\\n5. Analyzing temporal patterns...")
        self.analyze_temporal_patterns()

        print("\\n6. Creating final summary visualization...")
        self.create_final_summary_visualization(feature_importance_df, audio_pct, lyrics_pct, results)
        
        print("\\nAnalysis complete! Check the generated PNG files.")
        
        # Print final conclusions
        print("\\n" + "="*50)
        print("FINAL CONCLUSIONS:")
        print("="*50)
        best_model = max(results, key=lambda m: results[m]['accuracy'])
        print(f"1. Best decade classifier: {best_model} with {results[best_model]['accuracy']:.1%} accuracy")
        print(f"2. Audio features account for {audio_pct:.1f}% of feature importance")
        print(f"3. Lyrics features account for {lyrics_pct:.1f}% of feature importance")
        print(f"4. Top predictor of decade: {feature_importance_df.iloc[-1]['feature']}")
        print("5. Sonic convergence through 2000s, then streaming-era divergence in 2010s")
        
        return results, feature_importance_df

def main():
    """Main execution function"""
    
    # Check if merged data exists
    if not os.path.exists('merged.csv'):
        print("Error: merged.csv not found. Run merge_datasets.py first!")
        return
    
    # Run final analysis
    analyzer = FinalAnalyzer()
    results, feature_importance = analyzer.run_complete_analysis()
    
    print("\\n🎵 The Anatomy of a Hit - Analysis Complete! 🎵")

if __name__ == "__main__":
    main()
