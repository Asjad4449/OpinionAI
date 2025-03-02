from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from main import structured_debate
from semantic_analysis import analyze_debate  # Import your semantic analysis function
import base64
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    """ Renders the main page with the chat UI. """
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    opinion = data.get("opinion")

    print(f"Received opinion: {opinion}")  # Debugging print

    # Generate structured debate in JSON format
    debate = structured_debate(opinion)

    # Ensure Flask returns it as JSON
    return jsonify(debate)

@app.route("/semantic_analysis", methods=["POST"])
def semantic_analysis():
    debate_data = request.get_json()
    print("Received debate data for analysis:", debate_data)
    
    try:
        # Call your semantic analysis function directly with the debate data
        graph_paths = analyze_debate(debate_data)
        
        # Read and encode the generated graphs
        graphs = []
        for path in graph_paths:
            with open(path, 'rb') as f:
                graph_data = base64.b64encode(f.read()).decode('utf-8')
                graphs.append(f'data:image/png;base64,{graph_data}')
            # Clean up the temporary graph file
            os.remove(path)
            
        return jsonify({'graphs': graphs})
        
    except Exception as e:
        print(f"Error in semantic analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
