import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class SocialAnalyzer:
    def __init__(self):
        self.graph_paths = ['social_impact.png', 'platform_distribution.png']

    def analyze(self, debate_data):
        """
        Analyzes debate data for social media metrics.
        
        Args:
            debate_data (dict): The debate data containing pros and cons
            
        Returns:
            list: Paths to the generated graph files
        """
        pros = [item['argument'] for item in debate_data.get('pros', [])]
        cons = [item['argument'] for item in debate_data.get('cons', [])]
        
        # Example social media metrics (you can replace with real metrics)
        engagement_scores = self._calculate_engagement(pros + cons)
        platform_dist = self._analyze_platforms(pros + cons)
        
        # Generate and save graphs
        self._generate_impact_graph(engagement_scores)
        self._generate_platform_distribution(platform_dist)
        
        return self.graph_paths

    def _calculate_engagement(self, arguments):
        """Mock engagement calculation"""
        return {
            'shares': np.random.randint(100, 1000, len(arguments)),
            'likes': np.random.randint(500, 5000, len(arguments)),
            'comments': np.random.randint(50, 500, len(arguments))
        }

    def _analyze_platforms(self, arguments):
        """Mock platform distribution"""
        platforms = ['Twitter', 'Facebook', 'Reddit', 'Instagram']
        return Counter({p: np.random.randint(10, 100) for p in platforms})

    def _generate_impact_graph(self, engagement):
        plt.figure(figsize=(10, 6))
        x = range(len(engagement['shares']))
        plt.plot(x, engagement['shares'], 'b-', label='Shares')
        plt.plot(x, engagement['likes'], 'g-', label='Likes')
        plt.plot(x, engagement['comments'], 'r-', label='Comments')
        plt.title('Social Media Engagement Over Time')
        plt.xlabel('Arguments')
        plt.ylabel('Engagement Count')
        plt.legend()
        plt.savefig(self.graph_paths[0], bbox_inches='tight')
        plt.close()

    def _generate_platform_distribution(self, platform_dist):
        plt.figure(figsize=(10, 6))
        platforms = list(platform_dist.keys())
        counts = list(platform_dist.values())
        plt.pie(counts, labels=platforms, autopct='%1.1f%%')
        plt.title('Discussion Distribution Across Platforms')
        plt.savefig(self.graph_paths[1], bbox_inches='tight')
        plt.close() 