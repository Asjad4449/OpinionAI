import os
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from social_media import analyze_social_impact

class SocialAnalyzer:
    def __init__(self):
        self.output_dir = "social_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph_paths = [
            os.path.join(self.output_dir, 'social_impact.png'),
            os.path.join(self.output_dir, 'platform_distribution.png')
        ]

    def analyze(self, debate_data):
        """
        Quick social media analysis without visualizations.
        
        Args:
            debate_data (dict): The debate data containing topic
            
        Returns:
            dict: Analysis results
        """
        try:
            topic = debate_data.get('topic', '')
            result = analyze_social_impact(topic)
            
            if not result.get('success'):
                raise ValueError(result.get('error', 'Analysis failed'))
                
            return result['analysis']
            
        except Exception as e:
            print(f"Error in social analysis: {str(e)}")
            raise

    def _generate_social_graphs(self, data):
        """Generates visualization graphs from social analysis data."""
        # First graph - Social Impact
        plt.figure(figsize=(10, 6))
        sentiment_data = data['analysis']['emotional_data']
        metrics = ['Sentiment', 'Engagement', 'Impact']
        values = [
            sentiment_data['sentiment_score'],
            data['analysis']['metrics']['discussion_volume'] == 'High' and 0.8 or 0.5,
            data['analysis']['metrics']['controversy_level'] == 'High' and 0.8 or 0.5
        ]
        
        plt.bar(metrics, values, color=['#0d6efd', '#dc3545', '#198754'])
        plt.title('Social Media Impact Analysis')
        plt.ylabel('Score')
        plt.savefig(self.graph_paths[0], bbox_inches='tight', dpi=300)
        plt.close()

        # Second graph - Platform Distribution
        plt.figure(figsize=(10, 6))
        platforms = ['Reddit', 'Twitter', 'Other']
        # Dummy values for demonstration
        platform_values = [0.6, 0.8, 0.4]
        
        plt.bar(platforms, platform_values, color=['#ff4500', '#1da1f2', '#6c757d'])
        plt.title('Platform Discussion Distribution')
        plt.ylabel('Engagement Level')
        plt.savefig(self.graph_paths[1], bbox_inches='tight', dpi=300)
        plt.close() 