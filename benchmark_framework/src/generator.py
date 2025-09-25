import os
import json
from collections import defaultdict
from dotenv import load_dotenv
from datasets import load_dataset
import dspy
from tqdm import tqdm
import random
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import openai
import time

# configure logging at the top of your script
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data/output/dpr_llm_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

load_dotenv()

def get_model_info(model_name):
    if model_name == "llama-3-3-70b":
        api_base = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"
        model_id = "openai/meta-llama/llama-3-3-70b-instruct"
    elif model_name == "gpt-oss-120b":
        api_base = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b/v1"
        model_id = "openai/openai/gpt-oss-120b"
    elif model_name == "DeepSeek-V3":
        api_base = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3-h200/v1"
        model_id = "openai/deepseek-ai/DeepSeek-V3"
    elif model_name == "qwen-2-5-72b":
        api_base = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct/v1"
        model_id = "openai/Qwen/Qwen2.5-72B-Instruct"    
    elif model_name == "mixtral-8x22b":
        api_base = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01/v1"
        model_id = "openai/mistralai/mixtral-8x22B-instruct-v0.1"
    else:
        raise Exception(f"Unknown LLM {model_name}")

    return api_base, model_id

def configure_dspy_model(model_name):

    api_base, model_id = get_model_info(model_name)
    RITS_API_KEY = os.environ['RITS_API_KEY']
    llama3_70b_ins = dspy.LM(
        model=model_id,
        cache=True,
        max_tokens=4000,
        temperature=0,
        api_base=api_base,
        api_key=RITS_API_KEY,
        extra_headers={'RITS_API_KEY': RITS_API_KEY}
    )
    dspy.configure(lm=llama3_70b_ins)
    dspy.settings.configure(async_max_workers=10)


class QuestionAbstraction(dspy.Signature):
    """
    You are a Data Product Request Generator.

    # Context:
    Data Product is defined as a self-contained, reusable, and consumable data asset designed to deliver specific value to its users for data-driven use cases. A data product request is a high‑level specification of what data and analysis the user needs.

    You are given a cluster containing multiple tables.  
    Each table includes:  
    - a title (short description of the table)  
    - a list of column headers  
    - a set of user questions originally that can be answered from the given table.  

    Task:  
    Write one data product request that effectively represents the combined data needs expressed across questions and all given tables (data product cluster).

    Instructions:  
    - Do not copy or rephrase the input questions, the request should include the tables information in the cluster.
    - Identify distinct analytical goals and write one sentence per goal.  
    - Use a clear, professional tone suitable for real-world user requests, don't just use "analysis".  
    - Focus on general insights, comparisons, relationships, or patterns the data engineers should support.  

    Output format:  
    Return only the final data product request as plain text.

    Examples:  
    1. Gather data on hospital readmission rates for heart failure patients across different regions, and analyze which patient demographics or treatment protocols are most strongly associated with reduced readmission.  
    2. Compile data on the highest-grossing films of the past 15 years and analyze how factors such as genre, director, production budget, and release season contribute to box office performance.  
    3. Collect data showing changes in undergraduate admission rates at top U.S. public universities over the last decade, and assess how SAT scores, tuition, and diversity metrics influence these trends.
    4. Collect data that will allow queries on student and professor performance, including course satisfaction, grades, and demographics. It should also track student research capability, GPA, and course difficulty, as well as professor teaching ability and popularity. It should support evaluation of student success and professor effectiveness.
    5. Compile a dataset that will allow queries on customer spending habits and behavior. It should include information on payment methods, consumption patterns, and average prices paid. The data should also track changes in consumption over time and spending at specific locations, such as gas stations. This should allow for insights into customer segments and their financial activities.
    """

    cluster_info: list[dict] = dspy.InputField(
        desc="Cluster of tables with title, columns, and questions."
    )

    data_product_question: str = dspy.OutputField(
        desc="The data product request that captures the information needs or intent of all the specific questions and tables."
    )


def call_llm(cluster_info):
    q_cot = dspy.ChainOfThought(QuestionAbstraction)
    logging.info("LLM input: %s", cluster_info)
    # print(q_cot)
    llm_output = q_cot(
        cluster_info = cluster_info
    )
    dpr = llm_output.data_product_question
    reasoning = getattr(llm_output, "reasoning", None)

    # log the output
    logging.info("LLM output - DPR: %s", dpr)
    if reasoning:
        logging.info("LLM reasoning: %s", reasoning)

    return {
        "DPR": dpr,
        "reasoning": reasoning
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_corpus_maps(corpus):
    table_map = {}
    text_map  = {}
    table_text_map = {}

    for entry in corpus:
        tbl = entry.get("table", {})
        tid = tbl.get("table_id")
        if tid:
            table_map[tid] = {
                "uid":     tbl.get("uid", ""),
                "title":   tbl.get("title", ""),
                "columns": tbl.get("header", []),
            }
            table_text_map[tid] = []

        raw_text = entry.get("text", [])
        # normalize into a list of dicts
        if isinstance(raw_text, dict):
            text_items = [raw_text]
        elif isinstance(raw_text, list):
            text_items = raw_text
        else:
            # if it's e.g. a string or None, skip
            continue

        for txt in text_items:
            if not isinstance(txt, dict):
                continue
            uid = txt.get("uid")
            val = txt.get("value")
            if uid and isinstance(val, str):
                text_map[uid] = {"value": val}
                if tid:
                    table_text_map[tid].append(uid)

    return table_map, text_map, table_text_map


def process_single_cluster_worker(cluster, table_text_map, table_map, text_map):
    dpr_id    = cluster["dpr_id"]
    # cluster only contain table ids
    tables    = cluster.get("tables", [])
    text_uids = []
    for tbl in tables:
        tid = tbl["table_id"]
        text_uids.extend(table_text_map.get(tid, []))

    cluster_info = []
    for tbl in tables:
        tid = tbl["table_id"]
        if tid not in table_map:
            continue

        title = table_map[tid]["title"]
        # when the table title is empty, use the text info to fill in
        if not title and text_uids:
            first_uid = text_uids[0]
            title = text_map.get(first_uid, {}).get("value", "")

        ci = {
            "title":     title,
            "columns":   table_map[tid]["columns"],
            "questions": tbl.get("questions", []),
        }
        cluster_info.append(ci)

    llm_out  = call_llm(cluster_info)
    dpr_text = llm_out["DPR"]

    gt_table_uids = [
        table_map[t]["uid"]
        for t in [tbl["table_id"] for tbl in tables]
        if t in table_map
    ]
    gt_text_uids = [uid for uid in text_uids if uid in text_map]

    item = {
        "dpr_id":       dpr_id,
        "DPR":          dpr_text,
        "ground_truth": {
            "table": gt_table_uids,
            "text":  gt_text_uids
        }
    }
    return item


def main(args):
    filtered = load_json(args.raw_cluster_path)
    corpus   = load_json(args.corpus_path)
    table_map, text_map, table_text_map = build_corpus_maps(corpus)
    max_workers = args.max_workers
    model_name = args.model_name

    configure_dspy_model(model_name)

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers as needed
        # Submit all tasks
        future_to_cluster = {executor.submit(process_single_cluster_worker, cluster, table_text_map, table_map, text_map): cluster for cluster in filtered}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_cluster), total=len(future_to_cluster), desc="Generate DPRs by cluster"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logging.error("Error processing cluster: %s", e)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time/60:.4f} minutes for processing {len(results)} clusters.")

    with open(args.output_path, "w", encoding="utf-8") as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} DPRs to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_cluster_path", type=str, default="data/output/TATQA_filtered_clusters.json")
    parser.add_argument("--corpus_path", type=str, default="data/output/TATQA_corpus.json")
    parser.add_argument("--output_path", type=str, default="data/output/TATQA_dprs.jsonl")
    parser.add_argument("--model_name", type=str, default="llama-3-3-70b")
    parser.add_argument("--max_workers", type=int, default=100)
    args = parser.parse_args()

    main(args)
