import json

with open("D:/VolumeEStuff/sony/polyhouseOntology/qa_new_dataset.json") as f:
    data = json.load(f)

with open("converted_dataset.jsonl", "w") as out:
    for item in data["questions"]:
        formatted = f"### Question:\n{item['question']}\n### SPARQL:\n{item['sparql']}"
        out.write(json.dumps({"text": formatted}) + "\n")
