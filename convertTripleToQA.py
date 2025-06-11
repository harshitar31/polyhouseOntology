import rdflib
import json
from rdflib.namespace import RDF, RDFS, OWL

def format_uri(uri):
    if isinstance(uri, rdflib.term.URIRef):
        return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
    return str(uri)

def generate_qa_from_triple(s, p, o):
    subject = format_uri(s)
    predicate = format_uri(p)
    obj = format_uri(o)

    # Simple question templates
    if p == RDF.type:
        question = f"What type is {subject}?"
        answer = obj
    elif p == RDFS.label:
        question = f"What is the label of {subject}?"
        answer = obj
    elif p == RDFS.comment:
        question = f"What is the comment about {subject}?"
        answer = obj
    else:
        question = f"What is the {predicate} of {subject}?"
        answer = obj

    return {"question": question, "answer": answer}

def ttl_to_qa_json(ttl_path, output_json_path):
    g = rdflib.Graph()
    g.parse(ttl_path, format="ttl")

    qa_pairs = []

    for s, p, o in g:
        qa = generate_qa_from_triple(s, p, o)
        qa_pairs.append(qa)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(qa_pairs)} Q&A pairs written to {output_json_path}")

# Example usage
ttl_file = "terms.ttl"  # Replace with your .ttl file path
json_output_file = "qa_dataset.json"
ttl_to_qa_json(ttl_file, json_output_file)
