import anthropic
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Claude 3.7 Sonnet client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def fetch_references(statement, stance, counter=0):
    """
    Searches Google for relevant articles supporting the given stance.
    Fetches different sources for each argument by modifying the query slightly.
    Returns a list of references as (headline, link).
    """
    # Modify query slightly for each request to encourage diversity
    query_variations = [
        f"{statement} {stance} debate",
        f"{statement} {stance} statistics",
        f"{statement} {stance} academic research",
        f"{statement} {stance} legal analysis",
        f"{statement} {stance} pros and cons"
    ]
    
    query = query_variations[counter % len(query_variations)]  # Rotate query variations

    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": os.environ.get("SERPAPI_API_KEY"),
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    sources = [
        (res.get("title", "No Title"), res.get("link", "No Link"))
        for res in results.get("organic_results", [])[:3]
    ]
    

    return sources if sources else [("No relevant sources found.", "No link available")]

def generate_argument(statement, stance, debate_history, sources):
    """
    Generates an argument for or against the given statement while avoiding repetition.
    The response sounds natural, references sources, and introduces fresh points.
    """
    formatted_sources = "\n".join([f"[{headline}]({link})" for headline, link in sources])

    prompt = f"""
    Statement: "{statement}"
    
    You're debating this topic, taking the "{stance}" side. 
    Your goal is to **sound like a real human debater**, engaging in a natural and compelling discussion.

    **Debate history so far:**
    {debate_history}

    **References to support your stance:**
    {formatted_sources}

    ### Debate Guidelines:
    - **Avoid repeating points already mentioned in the debate history.** Introduce fresh perspectives!
    - **Directly counter the last argument.** Don't just restate previous claims—challenge them.
    - **Use a natural tone, like a real person debating.** Avoid robotic phrasing.
    - **Use analogies, rhetorical questions, or humor if appropriate.**
    - **If a source is provided, reference it naturally in your response.**

    Now, in a clear and engaging way, **write your argument in 3-4 sentences and less than 50 words.**
    """

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=200,
        system="You are an expert debater, skilled at making arguments sound natural, persuasive, and engaging while avoiding repetition.",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()

def structured_debate(statement, num_rounds=5):
    """
    Conducts a structured debate with alternating arguments.
    Returns a structured dictionary with `pros`, `cons`, and `sources`.
    """
    debate_history = {"pros": [], "cons": [], "sources": []}

    for i in range(num_rounds * 2):
        stance = "Pro" if i % 2 == 0 else "Con"
        point_num = (i // 2) + 1

        # Fetch **new** relevant sources for each argument
        sources = fetch_references(statement, stance, counter=i)

        # Convert existing arguments to plain text for history context
        history_text = "\n".join(
            [entry["argument"] for entry in debate_history["pros"]]
            + [entry["argument"] for entry in debate_history["cons"]]
        )

        # Generate argument
        argument = generate_argument(statement, stance, history_text, sources)

        # Store argument and sources in the dictionary
        if stance == "Pro":
            debate_history["pros"].append({"point": point_num, "argument": argument})
        else:
            debate_history["cons"].append({"point": point_num, "argument": argument})
        
        debate_history["sources"].append(sources)  # Store sources separately

    return debate_history


def write_debate_to_json(debate_history, filename="debate_results.json"):
    """
    Writes the debate_history dictionary to a JSON file in a structured format.
    """
    # Convert sources from list of tuples to list of dictionaries for JSON compatibility
    formatted_sources = []
    for source_list in debate_history["sources"]:
        formatted_sources.append([
            {"title": title, "link": link} for title, link in source_list
        ])

    # Convert debate history into a properly formatted JSON structure
    formatted_debate = {
        "pros": [{"point": i + 1, "argument": argument} for i, argument in enumerate(debate_history["pros"])],
        "cons": [{"point": i + 1, "argument": argument} for i, argument in enumerate(debate_history["cons"])],
        "sources": formatted_sources
    }

    # Save to JSON file
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(formatted_debate, json_file, indent=4, ensure_ascii=False)

    print(f"Debate saved successfully to {filename}")

def follow_up_chat(debate_history):
    """
    Allows the user to ask follow-up questions based on the debate.
    The AI uses the debate as context to generate responses.
    """
    while True:
        user_question = input("\nAsk a follow-up question (or type 'exit' to stop): ")
        if user_question.lower() == "exit":
            print("\nEnding follow-up chat.")
            break

        prompt = f"""
        The following debate has already taken place:

        {debate_history}

        Now, a user has asked the following follow-up question:
        "{user_question}"

        Answer the user's question **using the context of the debate** while providing additional insights if needed.
        Keep the response **concise, clear, and informative**.
        """

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            system="You are a knowledgeable assistant answering follow-up questions about a structured debate. Your responses should be informative and reference key points from the debate while adding relevant new insights.",
            messages=[{"role": "user", "content": prompt}]
        )

        print("\nAI Response:")
        print(response.content[0].text.strip())

#statement = "men should be allowed multiple wives"
#debate_result = structured_debate(statement)
#write_debate_to_json(debate_result)
# follow-up chat
#follow_up_chat(debate_result)