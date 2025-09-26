import os
import json
from collections import defaultdict
from dotenv import load_dotenv
from datasets import load_dataset
import dspy
from tqdm import tqdm
import random
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import argparse
import time
from llm_provider import setup_llm_provider

load_dotenv()

class EvaluationMetrics(BaseModel):
    quality: float = Field(..., description="Quality score (1 to 5).")
    clarity: float = Field(..., description="Clarity score (1 to 5).")

class TableInfo(BaseModel):
    table_title: str = Field(..., description="The title of the table.")
    table_columns: list[str] = Field(..., description="A list of column names of the table.")

# add 3 examples
class AlignmentEvaluation(dspy.Signature):
    """
    You are a domain expert who is knowledgeable in building data products and data catalogues. You have expertise in evaluating the quality of data product requests (DPRs)
    
    # Context:
    Data Product is defined as a self-contained, reusable, and consumable data asset designed to deliver specific value to its users for data-driven use cases. A data product request is a high‑level specification of what data and analysis the user needs.

    # Your task:
    Given (a) a data product request and (b) a list of tables with the table title and table columns, your task is to identify any tables that are NOT relevant to the data product request. A table is not relevant if it is not useful to answer the given data product request. You have to compare each table title and columns with the given data product request and decide if the given table is relevant to the data product request or not. The table does not have to fully match the data product request but it should atleast contain some relevant information to the data product request. If the table is not relevant, list the table in the output and provide a reasoning for your decision.
    """

    data_product_request: str = dspy.InputField(
        desc="The data product request specifying what data and analysis the user needs."
    )

    table_list: list[TableInfo] = dspy.InputField(
        desc="List of canidate tables along with table title and column names."
    )

    irrelevant_list: list[str] = dspy.OutputField(
        desc="List of table titles that are not aligned or not relevant for the given data product request."
    )


class DataProductRequestTextEvaluation(dspy.Signature):
    """
    You are LLMeval, an expert evaluator of the quality of data product requests (DPRs).
    Context:
    Data Product is defined as a self-contained, reusable, and consumable data asset designed to deliver specific value to its users for data-driven use cases. A data product request is a high‑level specification of what data and analysis the user needs.

    **Your task:** 

    Given:  
    - A data product request   

    For each DPR, assign a score from **1** to **5** for each of the 5 criteria below. Ensure scores are reasonable based on the Likert scale given below.

    **1. Quality - Level of abstraction (1-5):**
    Is the DPR phrased in a high-level of abstraction to cover some use case (not a factoid question), and in actionable manner suitable for guiding downstream data or analysis tasks?
    - 1 = Bad: Ambiguous, or not actionable request or a simple factoid question.
    - 2 = Weak: Partially contains either unclear or overly specific elements.
    - 3 = Adequate: Generally appropriate, but some awkward phrasing or lack of abstraction to cover a concrete use case.
    - 4 = Strong: Professional, mostly abstract to cover some use case and actionable, with minor lapses.
    - 5 = Ideal: Fully professional, written in a abstract way to cover some use case, concise, and actionable throughout.

    **2. Clarity (1–5):**
    Is the DPR clearly and unambiguously written, such that it can be readily understood and implemented?
    - 1 = Unclear: Hard to understand; ambiguous or confusing; incomplete or incomprehensible text.
    - 2 = Somewhat unclear: Parts are clear, but several ambiguities or awkward constructions of text.
    - 3 = Moderately clear: Mostly understandable; fluent with minor ambiguities.
    - 4 = Clear: Easy to follow, with only rare minor confusion.
    - 5 = Crystal clear: Unambiguous, fluent, well-written, immediately understandable throughout.

    """
    dpr: str = dspy.InputField(
        desc="The data product requests."
    )

    eval: EvaluationMetrics = dspy.OutputField(
        desc="Evaluation Results"
    )

def call_llm(dpr, gt, dimension):
    if dimension == "alignment":
        q_cot = dspy.ChainOfThought(AlignmentEvaluation)
        llm_output = q_cot(
            data_product_request=dpr,
            table_list = gt
        )
        irrelevant = llm_output.irrelevant_list 
        rel = 1 - (len(irrelevant)/len(gt))
        return {
            "response": {"rel": rel, "non-aligned-tables": irrelevant},
            "reasoning": getattr(llm_output, "reasoning", None)
        }
    else:
        q_cot = dspy.ChainOfThought(DataProductRequestTextEvaluation)
        llm_output = q_cot(
            dpr=dpr,
            table_list = gt
        )
        return {
            "response": llm_output.eval.model_dump(),
            "reasoning": getattr(llm_output, "reasoning", None)
        }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def map_table(corpus):
    table_map= {}
    for entry in corpus:
        tbl = entry.get("table", {})
        tid = tbl.get("uid")
        if not tid:
            continue
        table_map[tid] = TableInfo(
            table_title=tbl.get("title", ""),
            table_columns=tbl.get("header", [])
        )
    return table_map

def eval_single_dpr(dpr, corpus, dimension):
    table_ids = dpr["ground_truth"].get("table", [])
    table_map = map_table(corpus)
    table_infos = [table_map[tid] for tid in table_ids if tid in table_map]
    try:
        res = call_llm(dpr["DPR"], table_infos, dimension)
        eval_result = res['response']
        reasoning = res['reasoning']
        data = {'dpr_id': dpr["dpr_id"], 'DPR': dpr["DPR"], 'eval': eval_result, 'reasoning': reasoning}
    except Exception as ex:
        print(f"WARNING: Evaluation failure - {ex}")
        return None
    return data


def main(args):
    dprs   = load_jsonl(args.dprs_path)
    corpus = load_json(args.corpus_path)
    out_file = args.output_path
    max_workers = args.max_workers
    model_name = args.model_name
    eval_dimension = args.dimension

    setup_llm_provider(model_name)

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(eval_single_dpr, dpr, corpus, eval_dimension) for dpr in dprs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating DPRs"):
            result = future.result()
            if result is not None:
                results.append(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time/60:.4f} minutes for processing {len(results)} DPRs.")   

    with open(out_file, "w", encoding="utf-8") as fout:
        for result in results:
            json.dump(result, fout, ensure_ascii=False)
            fout.write('\n')
    print(f"Evaluations written to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/output/TATQA_corpus.json")
    parser.add_argument("--dprs_path", type=str, default="data/output/TATQA_dprs.jsonl")
    parser.add_argument("--output_path", type=str, default="data/output/TATQA_dprs_eval.jsonl")
    parser.add_argument("--model_name", type=str, default="llama-3-3-70b")
    parser.add_argument("--dimension", type=str, default="alignment")
    parser.add_argument("--max_workers", type=int, default=100)
    args = parser.parse_args()

    main(args)