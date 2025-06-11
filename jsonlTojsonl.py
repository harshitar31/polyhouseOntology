import json
import re

input_file = "D:/VolumeEStuff/sony/polyhouseOntology/newdataset.jsonl"  # Path to your input JSONL file
output_file = "D:/VolumeEStuff/sony/newdataset.jsonl"  # Path for the output JSONL file
valid_lines = []
import json

def fix_jsonl(input_path, output_path):
    fixed_lines = []
    buffer = ""

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            buffer += line.strip()
            try:
                # Try to parse the buffered line as JSON
                obj = json.loads(buffer)
                fixed_lines.append(json.dumps(obj))
                buffer = ""  # Reset buffer after a successful parse
            except json.JSONDecodeError:
                # If it's not complete JSON, keep buffering
                buffer += " "

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for fixed_line in fixed_lines:
            outfile.write(fixed_line + '\n')

    print(f"Fixed JSONL written to {output_path}")

# Usage:
fix_jsonl(input_file, output_file)
