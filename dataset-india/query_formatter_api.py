import json

# === CONFIGURATION ===
raw_response_file = "api_responses.json"
formatted_output_file = "testcases.json"

# === LOAD RAW RESPONSE ===
with open(raw_response_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === EXTRACT AND FORMAT CONTENT ===
formatted_testcases = []

for entry in raw_data:
    try:
        content = entry["response"]["openAIResponse"]["choices"][0]["message"]["content"]
        # Parse the content directly as JSON
        testcases =content
        formatted_testcases.extend(testcases)
    except Exception as e:
        print(f" Error processing entry: {e}")

# === SAVE FORMATTED TEST CASES ===
with open(formatted_output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_testcases, f, ensure_ascii=False, indent=4)

print(f" Saved {len(formatted_testcases)} test cases to {formatted_output_file}")
