import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from sentiment_analysis import analyze_debate_json, setup_nltk
import os

class SentimentAnalyzer:
    def __init__(self):
        self.output_dir = "debate_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph_paths = [
            os.path.join(self.output_dir, 'key_metrics.png'),
            os.path.join(self.output_dir, 'composite_scores.png'),
            os.path.join(self.output_dir, 'sentiment_radar.png')
        ]
        # Try to setup NLTK but don't raise if it fails
        setup_nltk()

    def analyze(self, debate_data):
        """
        Analyzes debate data and generates sentiment visualization graphs.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            tuple: (list of graph paths, analysis results dict)
        """
        try:
            # Try to setup NLTK again before analysis
            if not setup_nltk():
                print("Warning: NLTK setup failed, using default values")
                return self._get_default_results()
                
            visualization_paths, results = analyze_debate_json(debate_data)
            
            if visualization_paths is None or results is None:
                print("Warning: Analysis failed, using default values")
                return self._get_default_results()
            
            return visualization_paths, results
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return self._get_default_results()

    def _get_default_results(self):
        """Returns default results when analysis fails."""
        # Create empty graphs
        for path in self.graph_paths:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Data Available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            
        return (
            self.graph_paths,
            {
                "Pro Side": {
                    "Sentiment Score": 0,
                    "Emotional Intensity": 0,
                    "Expert References": 0,
                    "Practical Appeals": 0,
                    "Consistency Score": 0,
                    "Counter-argument Engagement": 0,
                    "Composite Score": 0
                },
                "Con Side": {
                    "Sentiment Score": 0,
                    "Emotional Intensity": 0,
                    "Expert References": 0,
                    "Practical Appeals": 0,
                    "Consistency Score": 0,
                    "Counter-argument Engagement": 0,
                    "Composite Score": 0
                }
            }
        )

    def _generate_sentiment_comparison(self, pro_sentiments, con_sentiments):
        """Generates bar graph comparing average sentiments"""
        plt.figure(figsize=(10, 6))
        plt.bar(['Pro Arguments', 'Con Arguments'], 
                [np.mean(pro_sentiments), np.mean(con_sentiments)],
                color=['#0d6efd', '#dc3545'])
        plt.title('Average Sentiment Analysis')
        plt.ylabel('Sentiment Score (-1 to 1)')
        plt.savefig(self.graph_paths[0], bbox_inches='tight')
        plt.close()

    def _generate_sentiment_flow(self, pro_sentiments, con_sentiments):
        """Generates line graph showing sentiment flow"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(pro_sentiments)), pro_sentiments, 'b-o', label='Pro')
        plt.plot(range(len(con_sentiments)), con_sentiments, 'r-o', label='Con')
        plt.title('Sentiment Flow Throughout Debate')
        plt.xlabel('Argument Number')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.savefig(self.graph_paths[1], bbox_inches='tight')
        plt.close() 