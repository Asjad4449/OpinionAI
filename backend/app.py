from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from debate_manager import DebateManager
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
debate_manager = DebateManager()

@app.route("/")
def index():
    """ Renders the main page with the chat UI. """
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        opinion = data.get("opinion")
        if not opinion:
            return jsonify({"error": "No opinion provided"}), 400
            
        debate = debate_manager.generate_debate(opinion)
        return jsonify(debate)
        
    except Exception as e:
        print(f"Error generating debate: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/sentiment_analysis", methods=["POST"])
def sentiment_analysis():
    try:
        debate_data = request.get_json()
        if not debate_data:
            return jsonify({"error": "No debate data provided"}), 400
            
        graphs = debate_manager.analyze_sentiment(debate_data)
        return jsonify({"graphs": graphs})
        
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/social_analysis", methods=["POST"])
def social_analysis():
    try:
        data = request.get_json()
        print("Received social analysis request data:", data)  # Debug log
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        topic = data.get('topic')
        print("Extracted topic:", topic)  # Debug log
        
        if not topic:
            return jsonify({"success": False, "error": "No topic provided"}), 400
            
        result = debate_manager.analyze_social(data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in social analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
