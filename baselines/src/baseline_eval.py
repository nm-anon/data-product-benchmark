import json
from sklearn.metrics import recall_score
import numpy as np
from collections import Counter
import argparse

MAX_NEG_RANK = 9999

def evaluate_dpr_result(pred_k, ground_truth, top_k=200, pr_top_k_list=[10, 20, 50, 100], hits_at_k_keys=[10,50,100,500,1000]):
    
    a_dpr_eval = {}
    for pr_top_k in pr_top_k_list:
        pred_ids = [x[0] for x in pred_k[:top_k]]

        pr_pred_ids = [x[0] for x in pred_k[:pr_top_k]]
        # p_y_true = [1 if id in ground_truth else 0 for id in pr_pred_ids]
        # p_y_pred = [1] * len(pr_pred_ids)

        # precision = precision_score(p_y_true, p_y_pred, zero_division=0)

        gt_not_in_pred = [1 for id in ground_truth if id not in pr_pred_ids]
        r_y_true = [1 if id in ground_truth else 0 for id in pr_pred_ids] + gt_not_in_pred
        r_y_pred = [1] * len(pr_pred_ids) + [0] * len(gt_not_in_pred)

        recall = recall_score(r_y_true, r_y_pred, zero_division=0)

        pred_ranks = [r+1 for r,id in enumerate(pred_ids) if id in ground_truth]
        pred_ranks_filtered = []
        for ind, r in enumerate(pred_ranks):
            # there are ind correct predictions before this rank, so apply filtering to inflate the rank
            pred_ranks_filtered.append(r-ind)
        pred_ranks_filtered += [MAX_NEG_RANK] * (len(ground_truth) - len(pred_ranks_filtered))

        a_dpr_eval[f'recall@{pr_top_k}'] = recall 
        # a_dpr_eval[f'precision@{pr_top_k}'] = precision
        
        a_dpr_eval[f'FullRecall@{pr_top_k}'] = 1 if recall == 1 else 0

    # print(pred_ranks_filtered)
    if pred_ranks_filtered:
        mrr = np.mean([1/r for r in pred_ranks_filtered])
    else:
        mrr = 0
    a_dpr_eval['mrr'] = mrr

    return a_dpr_eval

def evaluate_dprs(sys_results, dpr_to_gt, hits_at_k_keys=[10,50,100,500,1000]):

    eval_results = []
    for dpr_id in dpr_to_gt:
        dpr_result = {"dpr_id": dpr_id, "eval": {}}
        if dpr_id in sys_results:
            for rerank_type in ['rrf', 'bm25_only', 'vector_only']:
                dpr_result["eval"][rerank_type] = {}
                for d_type in ['table', 'text']:
                    a_dpr_sys_results = sys_results[dpr_id]['results'][rerank_type][d_type]
                    a_dpr_ground_truth = dpr_to_gt[dpr_id][d_type]
                    a_eval_result = evaluate_dpr_result(a_dpr_sys_results, a_dpr_ground_truth, hits_at_k_keys=hits_at_k_keys)
                    dpr_result["eval"][rerank_type][d_type] = a_eval_result
        eval_results.append(dpr_result)

    eval_summary = get_macro_average(eval_results)

    eval_data = {
        "global_eval" : eval_summary,
        "dpr_level_eval": eval_results
    }

    return eval_data

def get_macro_average(eval_results):
    total_count = 0
    rrf_text_counter = Counter()
    rrf_table_counter = Counter()
    rff_dp_full_count = 0
    bm25_text_counter = Counter()
    bm25_table_counter = Counter()
    bm25_dp_full_count = 0
    dense_text_counter = Counter()
    dense_table_counter = Counter()
    dense_dp_full_count = 0
    for eval_result in eval_results:
        total_count += 1
        eval = eval_result['eval']
        if 'rrf' in eval:
            if 'text' in eval['rrf']:
                add_results_to_counter(eval['rrf']['text'], rrf_text_counter)
            if 'table' in eval['rrf']:
                add_results_to_counter(eval['rrf']['table'], rrf_table_counter)
            if 'text' in eval['rrf'] and 'table' in eval['rrf']:
                if eval['rrf']['text']['FullRecall@100'] == 1 and eval['rrf']['table']['FullRecall@100'] == 1:
                    rff_dp_full_count += 1
        if 'bm25_only' in eval:
            if 'text' in eval['bm25_only']:
                add_results_to_counter(eval['bm25_only']['text'], bm25_text_counter)
            if 'table' in eval['bm25_only']:
                add_results_to_counter(eval['bm25_only']['table'], bm25_table_counter)
            if 'text' in eval['bm25_only'] and 'table' in eval['bm25_only']:
                if eval['bm25_only']['text']['FullRecall@100'] == 1 and eval['bm25_only']['table']['FullRecall@100'] == 1:
                    bm25_dp_full_count += 1
        if 'vector_only' in eval:
            if 'text' in eval['vector_only']:
                add_results_to_counter(eval['vector_only']['text'], dense_text_counter)
            if 'table' in eval['vector_only']:
                add_results_to_counter(eval['vector_only']['table'], dense_table_counter) 
            if 'text' in eval['vector_only'] and 'table' in eval['vector_only']:
                if eval['vector_only']['text']['FullRecall@100'] == 1 and eval['vector_only']['table']['FullRecall@100'] == 1:
                    dense_dp_full_count += 1

    eval_summary = {"total_dpr_count": total_count}
    eval_summary['rrf'] = {"text": avg_results_from_counter(rrf_text_counter, total_count), "table": avg_results_from_counter(rrf_table_counter, total_count), "dp_full_recall@100": rff_dp_full_count / total_count}
    eval_summary['bm25'] = {"text": avg_results_from_counter(bm25_text_counter, total_count), "table": avg_results_from_counter(bm25_table_counter, total_count), "dp_full_recall@100": bm25_dp_full_count / total_count}
    eval_summary['dense'] = {"text": avg_results_from_counter(dense_text_counter, total_count), "table": avg_results_from_counter(dense_table_counter, total_count), "dp_full_recall@100": dense_dp_full_count / total_count}
    return eval_summary

def add_results_to_counter(results, counter):
    for key in results:
        try:
            counter[key] += results[key]
        except Exception as e:
            print(f"Error adding result for key {key}: {e}")
            print(f"Results: {results[key]}")

def avg_results_from_counter(counter, total_count):
    avg_metrics = {}
    for key in counter:
        avg_metrics[key] = counter[key] / total_count
    return avg_metrics

def load_ground_truth(dpr_path):
    with open(dpr_path, "r", encoding="utf-8") as f:
        dprs = [json.loads(line) for line in f if line.strip()]
    print(f"{len(dprs)} DPR ground truth loaded.")

    dpr_to_gt = {}
    dpr_to_query = {}
    for dpr in dprs:
        dpr_id = dpr['dpr_id']
        query = dpr['DPR']
        text_gt = dpr['ground_truth']['synth_text']
        table_gt = dpr['ground_truth']['table']
        dpr_to_gt[dpr_id] = {"table": table_gt, "text": text_gt}
        dpr_to_query[dpr_id] = query

    return dpr_to_gt, dpr_to_query

def load_system_results(sys_results_path):
    with open(sys_results_path) as in_file:
        sys_results = json.load(in_file)
    sys_results = sys_results["results"]
    print(f"{len(sys_results)} system results loaded.")

    sys_results = {r["dpr_id"]: r for r in sys_results}
    return sys_results



def main(args):

    dpr_path = args.dpr
    sys_results_path = args.sys_output
    sys_eval_results_path = args.eval_output

    dpr_to_gt, dpr_to_query = load_ground_truth(dpr_path)
    sys_results = load_system_results(sys_results_path)

    eval_results = evaluate_dprs(sys_results, dpr_to_gt, hits_at_k_keys=[10])
    with open(sys_eval_results_path, "w") as out_file:
        json.dump(eval_results, out_file, ensure_ascii=False, indent=1) 


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DPR benchmark with Milvus")
    p.add_argument("--dpr",         type=str, default="data/output/dprs_final.json")
    p.add_argument("--sys_output",         type=str, default="data/output/HybridQA/HybridQA_results_mpnet.json")
    p.add_argument("--eval_output",         type=str, default="data/output/HybridQA/HybridQA_results_mpnet.json")

    args = p.parse_args()
    main(args)