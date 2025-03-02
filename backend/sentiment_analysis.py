import re
import argparse
import statistics
import json
import nltk
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import ssl

def setup_nltk():
    """Setup NLTK by downloading required resources and handling SSL certificate issues."""
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Basic required resources
        required_resources = [
            'vader_lexicon',
            'punkt',
            'stopwords'
        ]
        
        # Download resources
        for resource in required_resources:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
                return False
        
        # Simple verification - just check if we can create the analyzer
        try:
            sia = SentimentIntensityAnalyzer()
            return True
        except Exception as e:
            print(f"Error verifying VADER analyzer: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        return False

# Call setup at module import
setup_nltk()  # Remove the conditional raise

def analyze_sentiment(text):
    """Analyze sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def get_emotional_intensity(text):
    """Calculate emotional intensity based on adverbs and adjectives."""
    intensifiers = [
        'very', 'extremely', 'incredibly', 'absolutely', 'completely', 'truly',
        'deeply', 'strongly', 'profoundly', 'utterly', 'totally', 'highly'
    ]
    emotional_words = [
        'good', 'bad', 'terrible', 'wonderful', 'awful', 'excellent',
        'horrific', 'amazing', 'important', 'crucial', 'critical', 'essential'
    ]
    
    words = word_tokenize(text.lower())
    total_words = len(words)
    intensity_count = sum(1 for word in words if word in intensifiers or word in emotional_words)
    
    return intensity_count / total_words if total_words > 0 else 0

def get_key_terms(text, sentiment_type='positive'):
    """Extract key positive or negative terms from text."""
    sia = SentimentIntensityAnalyzer()
    words = word_tokenize(text.lower())
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_words = [word for word in words if word.isalpha() and word not in stopwords]
    
    if sentiment_type == 'positive':
        return [word for word in filtered_words if sia.polarity_scores(word)['pos'] > 0.5]
    else:  # negative
        return [word for word in filtered_words if sia.polarity_scores(word)['neg'] > 0.5]

def count_expert_references(text):
    """Count references to expert consensus or evidence."""
    expert_terms = [
        'research', 'study', 'evidence', 'expert', 'scientist', 'professional',
        'consensus', 'data', 'statistics', 'findings', 'report', 'analysis',
        'psychologist', 'educator', 'doctor', 'professor', 'academic', 'scholar'
    ]
    count = 0
    for term in expert_terms:
        count += len(re.findall(r'\b' + term + r'\w*\b', text.lower()))
    return count

def count_practical_appeals(text):
    """Count appeals to practical outcomes or real-world impacts."""
    practical_terms = [
        'impact', 'effect', 'result', 'outcome', 'consequence', 'benefit',
        'harm', 'improve', 'reduce', 'increase', 'decrease', 'change',
        'practical', 'reality', 'actually', 'implementation', 'application'
    ]
    count = 0
    for term in practical_terms:
        count += len(re.findall(r'\b' + term + r'\w*\b', text.lower()))
    return count

def calculate_consistency(points):
    """Calculate internal consistency of arguments."""
    # Simple approach: check for repeated key terms across points
    common_terms = Counter()
    for point in points:
        words = word_tokenize(point.lower())
        stopwords = nltk.corpus.stopwords.words('english')
        filtered_words = [word for word in words if word.isalpha() and word not in stopwords]
        point_terms = Counter(filtered_words)
        common_terms.update(point_terms.keys())
    
    # Count terms that appear in multiple points
    repeated_terms = sum(1 for term, count in common_terms.items() if count > 1)
    total_terms = len(common_terms)
    
    return repeated_terms / total_terms if total_terms > 0 else 0

def counter_argument_engagement(pro_points, con_points):
    """Measure how well each side engages with counter-arguments."""
    # Look for terms or phrases that indicate engagement with opposing views
    engagement_phrases = [
        'opponent', 'disagree', 'contrary', 'however', 'nevertheless', 'despite',
        'although', 'claim', 'argue', 'suggest', 'position', 'view', 'perspective',
        'criticism', 'critique', 'challenge', 'alternative', 'misunderstand'
    ]
    
    pro_engagement = 0
    con_engagement = 0
    
    for point in pro_points:
        for phrase in engagement_phrases:
            if phrase in point.lower():
                pro_engagement += 1
                break
    
    for point in con_points:
        for phrase in engagement_phrases:
            if phrase in point.lower():
                con_engagement += 1
                break
    
    pro_score = pro_engagement / len(pro_points) if pro_points else 0
    con_score = con_engagement / len(con_points) if con_points else 0
    
    return pro_score, con_score

def get_tone_progression(points):
    """Analyze how tone progresses across arguments."""
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(point)['compound'] for point in points]
    
    # Calculate slope of tone progression
    if len(scores) > 1:
        x = list(range(len(scores)))
        slope = 0
        if len(scores) > 1:
            # Simple linear regression slope calculation
            mean_x = sum(x) / len(x)
            mean_y = sum(scores) / len(scores)
            numerator = sum((x[i] - mean_x) * (scores[i] - mean_y) for i in range(len(x)))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            slope = numerator / denominator if denominator != 0 else 0
        return slope
    return 0

def create_visualization(results, topic, output_dir="debate_visualizations"):
    """Create visualizations of the debate analysis results."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Format safe filename from topic
    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
    
    # Set style for better looking plots
    plt.style.use('ggplot')
    
    # 1. Bar Chart: Key Metrics Comparison
    plt.figure(figsize=(12, 7))
    
    metrics = ['Sentiment Score', 'Emotional Intensity', 'Expert References', 
               'Practical Appeals', 'Consistency Score', 'Counter-argument Engagement']
    
    pro_values = [results['Pro Side'][metric] if isinstance(results['Pro Side'][metric], (int, float)) 
                 else 0 for metric in metrics]
    con_values = [results['Con Side'][metric] if isinstance(results['Con Side'][metric], (int, float)) 
                 else 0 for metric in metrics]
    
    # Normalize Expert References and Practical Appeals for better visualization
    max_expert_refs = max(pro_values[2], con_values[2])
    if max_expert_refs > 0:
        pro_values[2] /= max_expert_refs
        con_values[2] /= max_expert_refs
    
    max_practical_appeals = max(pro_values[3], con_values[3])
    if max_practical_appeals > 0:
        pro_values[3] /= max_practical_appeals
        con_values[3] /= max_practical_appeals
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    pro_bars = ax.bar(x - width/2, pro_values, width, label='Pro Side', color='#4CAF50', alpha=0.8)
    con_bars = ax.bar(x + width/2, con_values, width, label='Con Side', color='#F44336', alpha=0.8)
    
    ax.set_title(f'Debate Analysis: {topic}', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(pro_bars)
    autolabel(con_bars)
    
    plt.tight_layout()
    key_metrics_file = os.path.join(output_dir, f'{safe_topic}_key_metrics.png')
    plt.savefig(key_metrics_file, dpi=300)
    print(f"Key metrics visualization saved to {key_metrics_file}")
    plt.close()
    
    # 2. Composite Score Visualization
    plt.figure(figsize=(10, 6))
    
    scores = [results['Pro Side']['Composite Score'], results['Con Side']['Composite Score']]
    colors = ['#4CAF50', '#F44336']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(['Pro Side', 'Con Side'], scores, color=colors, alpha=0.8)
    
    # Highlight winner
    winner_idx = 0 if scores[0] > scores[1] else 1
    bars[winner_idx].set_alpha(1.0)
    bars[winner_idx].set_edgecolor('black')
    bars[winner_idx].set_linewidth(2)
    
    ax.set_title(f'Debate Composite Scores: {topic}', fontsize=16, pad=20)
    ax.set_ylabel('Composite Score (higher is better)')
    ax.set_ylim(0, 1)
    
    # Add labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Mark the winner
    winner_text = 'Winner' if abs(scores[0] - scores[1]) > 0.05 else 'Slight Edge'
    ax.text(winner_idx, scores[winner_idx] + 0.05, winner_text, 
            ha='center', va='bottom', fontweight='bold', color='black')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    composite_score_file = os.path.join(output_dir, f'{safe_topic}_composite_scores.png')
    plt.savefig(composite_score_file, dpi=300)
    print(f"Composite score visualization saved to {composite_score_file}")
    plt.close()
    
    # 3. Sentiment Analysis Radar Chart
    categories = ['Positive', 'Negative', 'Neutral', 'Emotional Intensity', 'Engagement']
    
    pro_metrics = [results['Pro Side']['Positive'], 
                  results['Pro Side']['Negative'], 
                  results['Pro Side']['Neutral'],
                  results['Pro Side']['Emotional Intensity'],
                  results['Pro Side']['Counter-argument Engagement']]
    
    con_metrics = [results['Con Side']['Positive'], 
                  results['Con Side']['Negative'], 
                  results['Con Side']['Neutral'],
                  results['Con Side']['Emotional Intensity'],
                  results['Con Side']['Counter-argument Engagement']]
    
    # Number of categories
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the last value to close the loop
    pro_metrics += pro_metrics[:1]
    con_metrics += con_metrics[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Draw the Pro side
    ax.plot(angles, pro_metrics, linewidth=1, linestyle='solid', label="Pro Side", color='#4CAF50')
    ax.fill(angles, pro_metrics, alpha=0.1, color='#4CAF50')
    
    # Draw the Con side
    ax.plot(angles, con_metrics, linewidth=1, linestyle='solid', label="Con Side", color='#F44336')
    ax.fill(angles, con_metrics, alpha=0.1, color='#F44336')
    
    # Add title and legend
    plt.title(f'Sentiment Analysis: {topic}', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    sentiment_radar_file = os.path.join(output_dir, f'{safe_topic}_sentiment_radar.png')
    plt.savefig(sentiment_radar_file, dpi=300)
    print(f"Sentiment radar visualization saved to {sentiment_radar_file}")
    plt.close()
    
    # Return paths to created visualizations
    return [key_metrics_file, composite_score_file, sentiment_radar_file]

def analyze_debate_json(debate_data):
    """Main function to analyze the debate from JSON data."""
    output_dir = "debate_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract pro and con points
        pro_points = []
        con_points = []
        
        # Extract points
        for point_data in debate_data.get('pros', []):
            if 'argument' in point_data:
                pro_points.append(point_data['argument'])
        
        for point_data in debate_data.get('cons', []):
            if 'argument' in point_data:
                con_points.append(point_data['argument'])
        
        if not pro_points or not con_points:
            print("Error: Could not extract valid debate points")
            return None, None
        
        # Combine all points for each side
        pro_text = " ".join(pro_points)
        con_text = " ".join(con_points)
        
        # Get sentiment scores
        pro_sentiment = analyze_sentiment(pro_text)
        con_sentiment = analyze_sentiment(con_text)
        
        # Get emotional intensity
        pro_intensity = get_emotional_intensity(pro_text)
        con_intensity = get_emotional_intensity(con_text)
        
        # Get expert references
        pro_expert_refs = count_expert_references(pro_text)
        con_expert_refs = count_expert_references(con_text)
        
        # Get practical appeals
        pro_practical_appeals = count_practical_appeals(pro_text)
        con_practical_appeals = count_practical_appeals(con_text)
        
        # Calculate consistency
        pro_consistency = calculate_consistency(pro_points)
        con_consistency = calculate_consistency(con_points)
        
        # Calculate counter-argument engagement
        pro_engagement, con_engagement = counter_argument_engagement(pro_points, con_points)
        
        # Get tone progression
        pro_tone_slope = get_tone_progression(pro_points)
        con_tone_slope = get_tone_progression(con_points)
        
        # Create results dictionary
        results = {
            "Pro Side": {
                "Sentiment Score": round(pro_sentiment['compound'], 2),
                "Positive": round(pro_sentiment['pos'], 2),
                "Negative": round(pro_sentiment['neg'], 2),
                "Neutral": round(pro_sentiment['neu'], 2),
                "Emotional Intensity": round(pro_intensity, 2),
                "Expert References": pro_expert_refs,
                "Practical Appeals": pro_practical_appeals,
                "Consistency Score": round(pro_consistency, 2),
                "Counter-argument Engagement": round(pro_engagement, 2),
                "Tone Progression": "Becoming more assertive" if pro_tone_slope > 0.05 else "Becoming less assertive" if pro_tone_slope < -0.05 else "Consistent tone",
                "Composite Score": round((pro_sentiment['compound'] + pro_engagement + pro_consistency) / 3, 2)
            },
            "Con Side": {
                "Sentiment Score": round(con_sentiment['compound'], 2),
                "Positive": round(con_sentiment['pos'], 2),
                "Negative": round(con_sentiment['neg'], 2),
                "Neutral": round(con_sentiment['neu'], 2),
                "Emotional Intensity": round(con_intensity, 2),
                "Expert References": con_expert_refs,
                "Practical Appeals": con_practical_appeals,
                "Consistency Score": round(con_consistency, 2),
                "Counter-argument Engagement": round(con_engagement, 2),
                "Tone Progression": "Becoming more assertive" if con_tone_slope > 0.05 else "Becoming less assertive" if con_tone_slope < -0.05 else "Consistent tone",
                "Composite Score": round((con_sentiment['compound'] + con_engagement + con_consistency) / 3, 2)
            }
        }
        
        # Generate visualizations using your existing create_visualization function
        debate_topic = "Debate Analysis"  # You can extract this from the debate data if available
        visualization_paths = create_visualization(results, debate_topic, output_dir)
        
        return visualization_paths, results
        
    except Exception as e:
        print(f"Error in analyze_debate_json: {str(e)}")
        return None, None

def sentiment_description(score):
    """Convert sentiment score to descriptive text."""
    if score >= 0.5:
        return "strongly positive"
    elif score >= 0.2:
        return "moderately positive"
    elif score > 0:
        return "mildly positive"
    elif score == 0:
        return "neutral"
    elif score >= -0.2:
        return "mildly negative"
    elif score >= -0.5:
        return "moderately negative"
    else:
        return "strongly negative"

def intensity_description(score):
    """Convert intensity score to descriptive text."""
    if score >= 0.6:
        return "very high"
    elif score >= 0.4:
        return "high"
    elif score >= 0.3:
        return "moderate"
    elif score >= 0.2:
        return "low"
    else:
        return "very low"

def consistency_description(score):
    """Convert consistency score to descriptive text."""
    if score >= 0.8:
        return "highly consistent"
    elif score >= 0.6:
        return "moderately consistent"
    elif score >= 0.4:
        return "somewhat consistent"
    else:
        return "inconsistent"

def engagement_description(score):
    """Convert engagement score to descriptive text."""
    if score >= 0.7:
        return "strongly engages opposing points"
    elif score >= 0.5:
        return "adequately addresses opposing points"
    elif score >= 0.3:
        return "minimally addresses opposing points"
    else:
        return "rarely addresses opposing points"

def get_relative_strengths(results, side1, side2):
    """Identify metrics where side1 outperforms side2."""
    strengths = []
    
    comparisons = [
        ("Expert References", "Expert consensus references"),
        ("Practical Appeals", "Practical outcome appeals"),
        ("Consistency Score", "Internal consistency"),
        ("Counter-argument Engagement", "Direct engagement with counter-arguments"),
        ("Emotional Intensity", "Emotional impact")
    ]
    
    for metric, description in comparisons:
        if results[side1][metric] > results[side2][metric]:
            strengths.append(description)
    
    # Add tone strength if applicable
    if side1 == "Pro Side" and results[side1]["Tone Progression"] == "Becoming more assertive":
        strengths.append("Gradually builds case with increasing conviction")
    elif side1 == "Con Side" and results[side1]["Tone Progression"] == "Consistent tone":
        strengths.append("Measured tone consistency")
    
    return strengths[:3]  # Return top 3 strengths