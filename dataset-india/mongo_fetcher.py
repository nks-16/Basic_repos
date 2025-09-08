from pymongo import MongoClient
from bson import ObjectId, json_util
import json

def fetch_filtered_messages_by_id(
    target_id,
):
    username = ""
    password = ""
    host = ""
    port = 27017
    auth_db = ""
    db_name = ""
    collection_name = ""
    uri = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_db}"
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    output_file = "filtered_messages.json"

    pipeline = [
        { "$match": { "orderItemUUID": target_id } },
        {
            "$project": {
                "messages": {
                    "$map": {
                        "input": {
                            "$filter": {
                                "input": "$messages",
                                "as": "msg",
                                "cond": {
                                    "$in": ["$$msg.role", ["user", "assistant"]]
                                }
                            }
                        },
                        "as": "filteredMsg",
                        "in": {
                            "role": "$$filteredMsg.role",
                            "content": "$$filteredMsg.content"
                        }
                    }
                }
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    if not results:
        raise ValueError("No matching document found.")
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, default=json_util.default, indent=4)

    print(f" Saved filtered messages for ID {target_id} to {output_file}")
    return results[0]["messages"]