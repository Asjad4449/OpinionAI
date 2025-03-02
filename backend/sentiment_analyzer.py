import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        self.graph_paths = ['sentiment_comparison.png', 'sentiment_flow.png']

    def analyze(self, debate_data):
        """
        Analyzes debate data and generates sentiment visualization graphs.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            list: Paths to the generated graph files
        """
        pros = [item['argument'] for item in debate_data.get('pros', [])]
        cons = [item['argument'] for item in debate_data.get('cons', [])]
        
        # Sentiment Analysis
        pro_sentiments = [TextBlob(text).sentiment.polarity for text in pros]
        con_sentiments = [TextBlob(text).sentiment.polarity for text in cons]
        
        # Generate and save graphs
        self._generate_sentiment_comparison(pro_sentiments, con_sentiments)
        self._generate_sentiment_flow(pro_sentiments, con_sentiments)
        
        return self.graph_paths

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