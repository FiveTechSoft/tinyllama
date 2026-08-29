import json
rec = {"id": "x", "prompt": "hola que tal", "tags": [], "response": "mundo estoy bien"}
with open("test1.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\n")
print("ok")