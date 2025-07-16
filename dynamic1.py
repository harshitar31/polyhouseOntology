import re
from rdflib import Graph
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1. Load Triples from TTL ===
def load_triples_from_ttl(file_path):
    g = Graph()
    g.parse(file_path, format="ttl")

    triples = []
    for s, p, o in g:
        s_str = s.split("/")[-1]
        p_str = p.split("/")[-1]
        o_str = o.split("/")[-1]
        triples.append((s_str, p_str, o_str))
    return triples

# === 2. Build Embedding Index ===
def build_index(triples, model_name="BAAI/bge-base-en-v1.5"):
    model = SentenceTransformer(model_name)
    texts = [f"passage: {s} {p} {o}" for s, p, o in triples]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, texts

# === 3. True Hybrid Retriever ===
def true_hybrid_retrieve(question, model, index, texts, triples, top_k=5, alpha=0.5):
    query = f"query: {question}"
    q_embed = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_embed, top_k)

    results = []
    question_keywords = set(re.findall(r'\w+', question.lower()))

    for rank, idx in enumerate(I[0]):
        triple = triples[idx]
        text = texts[idx]
        sem_score = -D[0][rank]  # negative L2 = similarity

        triple_words = set(text.lower().split())
        keyword_overlap = len(triple_words & question_keywords)
        boost = keyword_overlap / len(question_keywords) if question_keywords else 0

        final_score = alpha * sem_score + (1 - alpha) * boost
        results.append((final_score, triple))

    results.sort(reverse=True, key=lambda x: x[0])
    return results

# === 4. Prompt Construction ===
def format_prompt(question, top_triples):
    context_lines = [f"- {s} {p} {o}" for _, (s, p, o) in top_triples]
    context_text = "\n".join(context_lines)
    prompt = f"""You are a helpful assistant who is great at knowledge graphs and answering questions based on structured knowledge.
Only provide natural language answers based on the context.

Question: {question}
Context:
{context_text}

Answer in plain English:"""
    return prompt

# === 5. Answer Generation with Phi-2 ===
def generate_answer_phi2(prompt, tokenizer, model, device, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.split("Answer:")[-1].strip()

# === 6. Main ===
if __name__ == "__main__":
    print("🔄 Loading RDF triples from ontology.ttl...")
    ttl_file = "/content/drive/MyDrive/terms.ttl"
    triples = load_triples_from_ttl(ttl_file)
    print(f"✅ Loaded {len(triples)} triples")

    print("🔍 Building hybrid retriever index with BGE-base embeddings...")
    embed_model, index, texts = build_index(triples)

    print("🧠 Loading Phi-2 language model...")
    phi_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/phi-2")
    phi_model = AutoModelForCausalLM.from_pretrained(
        "/content/drive/MyDrive/phi-2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    phi_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi_model.to(device)

    print("💬 Ready! Ask questions or type 'exit' to quit.")
    while True:
        question = input("\n🔎 Question: ")
        if question.lower() in {"exit", "quit"}:
            break

        top_triples = true_hybrid_retrieve(question, embed_model, index, texts, triples, top_k=20)

        print("\n📚 Top Retrieved Triples:")
        for score, triple in top_triples:
            print(f"  - {triple} (score: {round(score, 3)})")

        prompt = format_prompt(question, top_triples)
        answer = generate_answer_phi2(prompt, phi_tokenizer, phi_model, device)

        print("\n🧠 Answer:")
        print(answer)
