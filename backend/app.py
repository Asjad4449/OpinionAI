from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from main import structured_debate

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


if __name__ == "__main__":
    app.run(debug=True)
