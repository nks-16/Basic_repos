import requests
import csv
import uuid
import time
import json

def run_api_sequence(user_query, order_item_uuid, num_iterations=1, csv_filename="query-sequence.csv"):
    # --- API ENDPOINTS ---
    FETCH_URL = ""
    CLEAR_URL = ""

    # --- HEADERS ---
    FETCH_HEADERS = {
              
    }

    CLEAR_HEADERS = {
    
    }

    # --- CSV SETUP ---
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["UUID", "OrderItemUUID", "UserQuery", "AIResponse"])

        for i in range(num_iterations):
            unique_id = str(uuid.uuid4())
            print(f"\n==> API HIT #{i+1} | UUID: {unique_id}")

            fetch_payload = {
                "message": user_query,
                "orderItemUUID": order_item_uuid
            }

            clear_payload = {
                "additionalDetails": {
                    "orderItemUUID": order_item_uuid
                }
            }

            # --- FETCH API CALL ---
            try:
                fetch_response = requests.post(
                    FETCH_URL,
                    headers=FETCH_HEADERS,
                    json=fetch_payload,
                    timeout=100
                )
                fetch_response.raise_for_status()
                response_json = fetch_response.json()

                ai_response = response_json  # You can refine this if needed
                writer.writerow([unique_id, order_item_uuid, user_query, ai_response])
                print(f"[FETCH] Status: {fetch_response.status_code}")
                print(f"[FETCH] AI Response: {ai_response}")

            except requests.exceptions.RequestException as err:
                writer.writerow([unique_id, order_item_uuid, user_query, f"Fetch Error: {err}"])
                print(f"[FETCH] Error: {err}")
                continue

            # --- CLEAR MEMORY API CALL ---
            try:
                clear_response = requests.post(
                    CLEAR_URL,
                    headers=CLEAR_HEADERS,
                    json=clear_payload,
                    timeout=10
                )
                clear_response.raise_for_status()
                print(f"[CLEAR] Status: {clear_response.status_code}")

            except requests.exceptions.RequestException as err:
                print(f"[CLEAR] Error: {err}")

            time.sleep(1)
def extract_query_texts(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        query_texts = [item["queryText"] for item in data if "queryText" in item]
        return query_texts    
    
if __name__ == "__main__":
    queries=extract_query_texts("testcases.json")
    for query in queries:
        run_api_sequence(query, "8051a7c93a35e006c4cac56701020200", num_iterations=1, csv_filename="query-sequence.csv")

