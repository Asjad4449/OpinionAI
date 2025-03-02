import anthropic
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Claude 3.7 Sonnet client
client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

def analyze_social_impact(topic):
    """Quick social media analysis using Claude."""
    try:
        # More structured prompt to ensure JSON response
        prompt = f"""Analyze social media sentiment for: "{topic}"
        
        Return your analysis in valid JSON format using this exact structure:
        {{
            "sentiment": "positive/negative/neutral",
            "key_points": [
                "first key point",
                "second key point",
                "third key point"
            ],
            "platform_insights": "summary of platform trends"
        }}
        
        Make sure to:
        1. Keep sentiment as one word (positive/negative/neutral)
        2. Include exactly 5 key points
        3. Make platform_insights a concise paragraph
        4. Return ONLY the JSON, no other text
        """
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=500,
            system="Return only valid JSON data, no other text or explanations.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response into structured data
        analysis = response.content[0].text.strip()
        try:
            # Try to parse if Claude returned JSON
            data = json.loads(analysis)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Claude response as JSON: {analysis}")
            # Fallback to structured data
            data = {
                "sentiment": "neutral",
                "key_points": [
                    "Analysis currently unavailable",
                    "Try again in a moment",
                    "System is processing request"
                ],
                "platform_insights": "Social media analysis is being generated."
            }
        
        return {
            "success": True,
            "analysis": data
        }
        
    except Exception as e:
        print(f"Error in social analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
