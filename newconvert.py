import rdflib
import json
import os
import re
from rdflib.namespace import RDF

# === Format URI for human readability ===
def format_label(uri):
    if isinstance(uri, rdflib.term.URIRef):
        label = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        return re.sub(r'_', ' ', label)
    return str(uri)

# === Generate readable answer from list of URIs ===
def generate_answer(class_name, uris):
    labels = [format_label(uri) for uri in uris]
    if len(labels) == 1:
        return f"The {class_name.lower()} present is {labels[0]}."
    elif len(labels) == 2:
        return f"The {class_name.lower()}s present are {labels[0]} and {labels[1]}."
    else:
        return f"The {class_name.lower()}s present are {', '.join(labels[:-1])}, and {labels[-1]}."

# === Generate QA pairs for rdf:type classes ===
def generate_qa_by_class(g):
    qa_list = []
    classes = set(o for s, p, o in g if p == RDF.type)

    for class_uri in classes:
        class_name = format_label(class_uri)

        sparql_query = f"""
        SELECT ?instance WHERE {{
            ?instance rdf:type <{class_uri}> .
        }}
        """

        results = list(g.query(sparql_query))
        if results:
            instances = [row[0] for row in results]
            qa_list.append({
                "question": f"What are the {class_name.lower()}s present?",
                "sparql": sparql_query.strip(),
                "result": str(results),
                "answer": generate_answer(class_name, instances)
            })

    return qa_list

# === Append Mode JSON Output ===
def ttl_to_class_qa_json(ttl_path, output_json_path):
    g = rdflib.Graph()
    g.parse(ttl_path, format="ttl")
    g.bind("rdf", RDF)

    new_qa_pairs = generate_qa_by_class(g)

    # Load existing questions if present
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "questions" in data:
            data["questions"].extend(new_qa_pairs)
        else:
            data["questions"] = new_qa_pairs
    else:
        data = {"questions": new_qa_pairs}

    # Write back merged data
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Appended {len(new_qa_pairs)} new Q&A pairs to {output_json_path}")

# === Example usage ===
ttl_file = "terms.ttl"  # Change this
json_output_file = "qa_new_dataset.json"
ttl_to_class_qa_json(ttl_file, json_output_file)
