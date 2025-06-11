import json

def convert_qa_to_jsonl(input_path, output_path, system_prompt="You are a helpful assistant."):
    with open(input_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for pair in qa_data:
            question = pair.get("question")
            answer = pair.get("answer")
            if question and answer:
                chat_obj = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
                out_file.write(json.dumps(chat_obj, ensure_ascii=False) + "\n")

    print(f"✅ JSONL file written to {output_path} with {len(qa_data)} entries.")

# === USAGE ===
if __name__ == "__main__":
    input_file = "qa_dataset.json"       # Your original Q&A file
    output_file = "chat_dataset.jsonl"          # Output for fine-tuning
    convert_qa_to_jsonl(input_file, output_file)
