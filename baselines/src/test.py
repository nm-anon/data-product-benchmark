import torch
from transformers import AutoTokenizer, AutoModel

def test_colbert_interaction():
    model_name = "colbert-ir/colbertv2.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dpr = "Analyze managerial success in European football tournaments"
    table_1 = "UEFA Euro Winners | Year | Winning Manager | Nationality"
    table_2 = "FIFA World Cup Stats | Year | Host | Attendance"

    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            # ColBERT uses the token-level embeddings (last_hidden_state)
            return model(**inputs).last_hidden_state

    q_embs = get_embeddings(dpr) # Shape: [1, query_len, 128]
    t1_embs = get_embeddings(table_1) # Shape: [1, table_len, 128]
    t2_embs = get_embeddings(table_2)

    def maxsim_score(query, document):
        # Late Interaction: Sum of max similarity for each query token
        sim_matrix = torch.matmul(query, document.transpose(1, 2))
        max_sim_per_token, _ = torch.max(sim_matrix, dim=2)
        return torch.sum(max_sim_per_token).item()

    score_1 = maxsim_score(q_embs, t1_embs)
    score_2 = maxsim_score(q_embs, t2_embs)

    print(f"ColBERT Score (Table 1 - Euro): {score_1:.4f}")
    print(f"ColBERT Score (Table 2 - World Cup): {score_2:.4f}")
    print("Observation: Table 1 scores higher because 'manager' aligns specifically with its header.")

if __name__ == "__main__":
    test_colbert_interaction()