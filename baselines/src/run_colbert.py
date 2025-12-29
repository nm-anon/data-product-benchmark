import json
import os
import argparse
from tqdm import tqdm
from ragatouille import RAGPretrainedModel

def prepare_corpus(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    documents = []
    doc_ids = []
    
    # Extract tables and text as unique documents
    seen_ids = set()
    for entry in corpus_data:
        # Table Representation
        t_uid = entry["table"]["uid"]
        if t_uid not in seen_ids:
            table_rep = f"{entry['table']['title']} | " + " | ".join(entry['table']['header'])
            documents.append(table_rep)
            doc_ids.append(t_uid)
            seen_ids.add(t_uid)
            
        # Text Representation
        for text_passage in entry["synth_text"]:
            p_uid = text_passage['uid']
            if p_uid not in seen_ids:
                content = f"Title: {text_passage['title']}\nContent: {text_passage['text']}"
                documents.append(content)
                doc_ids.append(p_uid)
                seen_ids.add(p_uid)
                
    return documents, doc_ids

def main(args):
    # 1. Initialize official ColBERT checkpoint via RAGatouille
    # This uses the native colbert-ai backend with residual compression
    print(f"Loading ColBERT model: {args.model}")
    rag = RAGPretrainedModel.from_pretrained(args.model)

    # 2. Prepare Data
    documents, doc_ids = prepare_corpus(args.corpus)
    print(f"Indexing {len(documents)} documents natively...")

    # 3. Native Indexing (IVF + Residual Compression)
    # This creates a local '.ragatouille/' directory with the native index
    index_path = rag.index(
        collection=documents,
        document_ids=doc_ids,
        index_name=args.index_name,
        max_document_length=256,
        split_documents=False
    )


    with open(args.dpr, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    query_results = []
    print("Executing native MaxSim search...")
    for q in tqdm(queries, desc="Retrieving"):
        # native search() handles the query augmentation [Q] and MaxSim logic
        raw_results = rag.search(q["DPR"], k=args.top_k)
        
        # Format results to match your baseline for easier evaluation
        formatted_results = [
            {"uuid": r["document_id"], "score": r["score"]} for r in raw_results
        ]
        
        query_results.append({
            "dpr_id": q["dpr_id"],
            "query": q["DPR"],
            "results": {"colbert_native": formatted_results}
        })

    output_summary = {
        "dataset": args.dataset,
        "model": args.model,
        "results": query_results
    }
    
    with open(args.output_path, "w", encoding="utf-8") as out_file:
        json.dump(output_summary, out_file, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default="data/TATQA_corpus.json")
    p.add_argument("--dpr", type=str, default="data/TATQA_test.jsonl")
    p.add_argument("--dataset", type=str, default="TATQA")
    p.add_argument("--model", type=str, default="colbert-ir/colbertv2.0")
    p.add_argument("--index_name", type=str, default="standalone_colbert_index")
    p.add_argument("--output_path", type=str, default="results_colbert_native.json")
    p.add_argument("--top_k", type=int, default=100)
    args = p.parse_args()
    main(args)
    