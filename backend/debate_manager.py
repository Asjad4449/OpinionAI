import os
import base64
from main import structured_debate
from sentiment_analyzer import SentimentAnalyzer
from social_analyzer import SocialAnalyzer
import threading

class DebateManager:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.social_analyzer = SocialAnalyzer()
        self.social_analysis_cache = {}  # Cache for social analysis results

    def generate_debate(self, opinion):
        """
        Generates a structured debate from an opinion and starts social analysis.
        
        Args:
            opinion (str): The user's opinion to debate
            
        Returns:
            dict: Structured debate with pros and cons
        """
        # Start social analysis in background
        thread = threading.Thread(target=self._background_social_analysis, args=(opinion,))
        thread.start()
        
        # Generate and return debate
        return structured_debate(opinion)

    def _background_social_analysis(self, topic):
        """Runs social analysis in background and caches results."""
        try:
            debate_data = {"topic": topic}
            graphs = self.social_analyzer.analyze(debate_data)
            self.social_analysis_cache[topic] = {"graphs": graphs}
        except Exception as e:
            print(f"Error in background social analysis: {str(e)}")
            self.social_analysis_cache[topic] = {"error": str(e)}

    def analyze_sentiment(self, debate_data):
        """
        Performs sentiment analysis on debate data and returns encoded graphs and results.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            dict: Contains base64 encoded graphs and analysis results
        """
        # Generate graphs and get analysis results
        graph_paths, analysis_results = self.sentiment_analyzer.analyze(debate_data)
        
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
            
        return {
            'graphs': graphs,
            'analysis': analysis_results
        }

    def analyze_social(self, debate_data):
        """
        Returns social media analysis results.
        
        Args:
            debate_data (dict): The debate data containing topic
            
        Returns:
            dict: Contains analysis results
        """
        try:
            analysis = self.social_analyzer.analyze(debate_data)
            return {
                "success": True,
                "data": analysis
            }
        except Exception as e:
            print(f"Error in social analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 