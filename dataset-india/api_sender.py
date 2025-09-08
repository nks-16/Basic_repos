import json
import requests

def send_context_and_save_response(
    json_content,
):
    response_file_path = "api_responses.json"
    api_url = "llm-apiendpoint"  # Fill this in
    username = ""  # Fill this in
    password = ""  # Fill this in
    api_id = 40
    # Load context from file
    data =  json_content

    context_messages = data
    # Build request payload
    payload = {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Senior QA Engineer and domain expert in understanding and expanding user requirements. You are tasked with generating all possible queries that are similar in context to a given user query in english and spanish.\n\nInstructions:\n1. Carefullyrehensive list of all possible queries that:\n   - Maintain the same context and meaning as the original query.\n   - Are phrased differently to reflect natural variations analyze the given conversation and identify the core context and intent of the original user query.\n2. Generate a comp in how different users might ask the same thing.\n   - Include positive, negative, and edge-case variations.\n   - Follow the same logical conversational flow and tone as the original query.\n3. Ensure the queries are realistic, relevant, and free of unrelated topics.\n4. Each query should be short, clear, and unambiguous.\n\nOutput Format Requirements:\n- The output must be in valid JSON format with no text outside the JSON.\n- Use the following structure exactly:\n```json\n\n  {\n    \"queryID\": \"Q-001\",\n    \"queryText\": \"The rephrased query here\"\n]```  }\n]\n- All queries must be inside a single JSON array.\n- Use unique sequential IDs (Q-001, Q-002, Q-003, etc.).\n\nGoal:\nTo produce a structured JSON list of all possible queries that align with the intent of the original query while maintaining clarity, coverage, and natural conversational flow.Give atlest 10 english queries.",
                }
            ] + context_messages
        },
        "username": username,
        "password": password,
        "api": api_id
    }

    # Send POST request
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=payload)

    # Handle response
    if response.status_code == 200:
        result = response.json()

        with open(response_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f" Response appended to {response_file_path}")
        return result
    else:
        error_msg = f" Error {response.status_code}: {response.text}"
        print(error_msg)
        raise Exception(error_msg)
