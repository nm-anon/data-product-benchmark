import argparse
import re
import os
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

def load_table_id_to_uuid(base_dir, dataset, split):
    table_title_to_ids = defaultdict(list)
    file_path = base_dir + f"{dataset}_{split}_corpus.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for table_data in data:
        table_uuid = table_data['table']['uid']
        table_title = table_data['table']['title']
        table_title_to_ids[table_title].append(table_uuid)
    return table_title_to_ids

def load_dprs(base_dir, dataset, split):
    dpr_id_to_dpr = dict()
    pattern = f"{dataset}_{split}_dprs-(.+?)\.jsonl"
    pattern = re.compile(pattern)
    for fname in os.listdir(base_dir):
        match = pattern.match(fname)
        if not match:
            continue
        dpr_llm = match.groups()[0]
        file_path = os.path.join(base_dir, fname)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    dpr_id = f"{data['dpr_id']}--{dpr_llm}"
                    data['dpr_id'] = dpr_id
                    dpr_id_to_dpr[dpr_id] = data
    return dpr_id_to_dpr 

def load_dpr_eval_files(base_dir, dataset, dimension, split):
    results = []
    pattern = f"{dataset}_{split}_dpr_eval_{dimension}_g_(.+?)_e_(.+?)\.jsonl"
    pattern = re.compile(pattern)
    for fname in os.listdir(base_dir):
        match = pattern.match(fname)
        if not match:
            continue
        llm1, llm2 = match.groups()
        file_path = os.path.join(base_dir, fname)

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        results.append({"dpr_llm": llm1, "eval_llm": llm2, "data": data})
    
    return results

def reorg_data(alignment_results, dimension):
    set_llms = set()
    cluster_to_dprs = defaultdict(dict)

    for item in alignment_results:
        dpr_llm = item['dpr_llm']
        eval_llm = item['eval_llm']
        set_llms.add(dpr_llm)
        set_llms.add(eval_llm)

        for dpr_eval in item['data']:
            dpr_id = dpr_eval['dpr_id']
            dpr = dpr_eval['DPR']

            if dimension == "alignment":
                eval = dpr_eval['eval']['rel']
                non_algn = dpr_eval['eval']['non-aligned-tables']
                reasoning = dpr_eval['reasoning']

                cluster_info = cluster_to_dprs[dpr_id]
                if dpr_llm in cluster_info:
                    cluster_info[dpr_llm]['eval'][eval_llm] = {"eval": eval, "non_align": non_algn, "reason": reasoning}
                else:
                    cluster_info[dpr_llm] = {"dpr": dpr, "eval": {eval_llm: {"eval": eval, "non_align": non_algn, "reason": reasoning}}}
            elif dimension == "dpr_clarity":
                quality = dpr_eval['eval']['quality']
                clarity = dpr_eval['eval']['clarity']
                reasoning = dpr_eval['reasoning']

                cluster_info = cluster_to_dprs[dpr_id]
                if dpr_llm in cluster_info:
                    cluster_info[dpr_llm]['eval'][eval_llm] = {"quality": quality, "clarity": clarity, "reason": reasoning}
                else:
                    cluster_info[dpr_llm] = {"dpr": dpr, "eval": {eval_llm: {"quality": quality, "clarity": clarity, "reason": reasoning}}}
            else:
                raise Exception("Unknown document")

    print(f"{dimension} LLMs - {set_llms}")
    return cluster_to_dprs


def main(args):
    base_dir_path = args.base_dir
    dataset = args.dataset
    split = args.split
    alignment_cut_off_threshold = args.alignment_cut_off

    print(f"\n\n Dataset: {dataset}, Split: {split}")

    table_title_to_ids = load_table_id_to_uuid(base_dir_path, dataset, split)
    print(f"table title to id size: {len(table_title_to_ids)}")

    dpr_id_to_dpr = load_dprs(base_dir_path, dataset, split)
    print(f"dpr id to dpr size:  {len(dpr_id_to_dpr)}")


    dimension = "alignment"
    alignment_results = load_dpr_eval_files(base_dir_path, dataset, dimension, split)
    print(f"{len(alignment_results )} alignment files loaded!")

    cluster_to_dprs = reorg_data(alignment_results, dimension)

    dpr_id_score = dict()
    dpr_id_to_non_align = dict()

    for c_id in cluster_to_dprs:
        for dpr_llm in cluster_to_dprs[c_id]:
            dpr_id = f"{c_id}--{dpr_llm}"
            llm_dpr_data = cluster_to_dprs[c_id][dpr_llm]
            total_score = 0
            total_count = 0
            non_align_counter = Counter()
            for eval_llm in llm_dpr_data['eval']:
                data = llm_dpr_data['eval'][eval_llm]
                total_score += data['eval']
                total_count += 1
                for tbl in data['non_align']:
                    non_align_counter[tbl] += 1
            avg_score = total_score / total_count
            dpr_id_score[dpr_id] = avg_score
            dpr_id_to_non_align[dpr_id] = [tbl_id for tbl_id, count in non_align_counter.most_common() if count > 2]
    
    print(f"Filtering using alignment cutoff threshold: {alignment_cut_off_threshold}")
    to_be_removed = [dpr_id for dpr_id, score in dpr_id_score.items() if score < alignment_cut_off_threshold]
    print(f"  {len(to_be_removed)} DPRs to be removed out of {len(dpr_id_score)}")

    tbl_to_remove_counts  =  [len(dpr_id_to_non_align[dpr_id]) for dpr_id in dpr_id_to_non_align] 
    dpr_at_least_1 = [1 for tbl_count in tbl_to_remove_counts if tbl_count > 0]
    print(f"  at least one table removed from ground truth of a DPR: {sum(dpr_at_least_1)}")
    print(f"  mean number of tables removed from a DPR: {np.mean(tbl_to_remove_counts)}")

    dimension = "dpr_clarity"
    clairity_results = load_dpr_eval_files(base_dir_path, dataset, dimension, split)
    print(f"{len(alignment_results )} dpr_clarity files loaded!")

    cluster_to_dprs = reorg_data(clairity_results, dimension)

    quality_low_count = Counter()
    clarity_low_count = Counter()

    for c_id in cluster_to_dprs:
        for dpr_llm in cluster_to_dprs[c_id]:
            dpr_id = f"{c_id}--{dpr_llm}"
            data = cluster_to_dprs[c_id][dpr_llm]
            dpr_quality_low_count = 0
            dpr_clarity_low_count = 0
            for eval_llm in data['eval']:
                eval_data = data['eval'][eval_llm]
                if eval_data['quality'] < 3:
                    dpr_quality_low_count += 1
                if eval_data['clarity'] < 3:
                    dpr_clarity_low_count += 1
            quality_low_count[dpr_id] = dpr_quality_low_count
            clarity_low_count[dpr_id] =dpr_clarity_low_count

    clarity_low_set = {dpr_id for dpr_id, count in clarity_low_count.most_common() if count >= 2}
    print(f"clarity low set size: {len(clarity_low_set)}")

    quality_low_set = {dpr_id for dpr_id, count in quality_low_count.most_common() if count >= 2}
    print(f"quality low set size: {len(quality_low_set)}")

    all_low_set = clarity_low_set.union(quality_low_set)
    print(f"total low set size: {len(all_low_set)}")

    filtered_dprs = []
    empty_gt_count = 0
    for dpr_id in dpr_id_to_dpr:
        dpr_data = dpr_id_to_dpr[dpr_id]
        if dpr_id in all_low_set:
            continue
        if dpr_id in to_be_removed:
            continue
        non_aligns = dpr_id_to_non_align[dpr_id]
        gt_tbl_ids = dpr_data['ground_truth']['table']
        orig_len = len(gt_tbl_ids)
        for non_align_tbl in non_aligns:
            if non_align_tbl in table_title_to_ids:
                table_id_list = table_title_to_ids[non_align_tbl]
                for table_id in table_id_list:
                    if table_id in gt_tbl_ids:
                        gt_tbl_ids.remove(table_id)
        if len(gt_tbl_ids) == 0:
            empty_gt_count += 1
            continue
        dpr_data['ground_truth']['table'] = gt_tbl_ids
        filtered_dprs.append(dpr_data)

    print(f"number of DPRs with empty ground truth after filtering: {empty_gt_count}")
    print(f"size of the final filtered DPRs: {len(filtered_dprs)}")

    output_path = base_dir_path + f"{dataset}_{split}_dprs-all.jsonl"
    with open(output_path, "w") as out_file:
        for item in filtered_dprs:
            jsonl = json.dumps(item, ensure_ascii=False)
            out_file.write(jsonl+"\n")
    print(f"Final filtered DPRs written to: {output_path}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data/output/HybridQA/train/")
    parser.add_argument("--dataset", type=str, default="HybridQA")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--alignment_cut_off", type=float, default=0.5)
    args = parser.parse_args()

    main(args)