import os
import base64
from main import structured_debate
from sentiment_analyzer import SentimentAnalyzer
from social_analyzer import SocialAnalyzer

class DebateManager:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.social_analyzer = SocialAnalyzer()

    def generate_debate(self, opinion):
        """
        Generates a structured debate from an opinion.
        
        Args:
            opinion (str): The user's opinion to debate
            
        Returns:
            dict: Structured debate with pros and cons
        """
        return structured_debate(opinion)

    def analyze_sentiment(self, debate_data):
        """
        Performs sentiment analysis on debate data and returns encoded graphs.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            list: Base64 encoded graph images
        """
        # Generate graphs
        graph_paths = self.sentiment_analyzer.analyze(debate_data)
        
        # Convert graphs to base64
        graphs = []
        try:
            for path in graph_paths:
                with open(path, 'rb') as f:
                    graph_data = base64.b64encode(f.read()).decode('utf-8')
                    graphs.append(f'data:image/png;base64,{graph_data}')
                # Clean up temporary files
                os.remove(path)
        except Exception as e:
            print(f"Error processing graphs: {str(e)}")
            # Clean up any remaining files
            for path in graph_paths:
                if os.path.exists(path):
                    os.remove(path)
            raise
            
        return graphs 

    def analyze_social(self, debate_data):
        """
        Performs social media analysis on debate data and returns encoded graphs.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            list: Base64 encoded graph images
        """
        # Generate graphs
        graph_paths = self.social_analyzer.analyze(debate_data)
        
        # Convert graphs to base64
        graphs = []
        try:
            for path in graph_paths:
                with open(path, 'rb') as f:
                    graph_data = base64.b64encode(f.read()).decode('utf-8')
                    graphs.append(f'data:image/png;base64,{graph_data}')
                # Clean up temporary files
                os.remove(path)
        except Exception as e:
            print(f"Error processing graphs: {str(e)}")
            # Clean up any remaining files
            for path in graph_paths:
                if os.path.exists(path):
                    os.remove(path)
            raise
            
        return graphs 