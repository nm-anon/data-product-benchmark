# Data Product Benchmark

The source code and data associated with the WSDM 2026 submission titled "From Factoid Questions to Data Product Requests: Benchmarking Data Product Discovery over Tables and Text". This repo is for review purposes, the benchmark framework and baseline code will be available in GitHub and the benchmark data will be uploaded to HuggingFace for the community.

## Structure of the repo
```
benchmark_data/
├── ConvFinQA/
│   ├── ConvFinQA_corpus.json        # text + table corpora
│   ├── ConvFinQA_dev.jsonl          # DPRs + ground truth DPs
│   ├── ConvFinQA_test.jsonl
│   └── ConvFinQA_train.jsonl
├── HybridQA/
│   ├── HybridQA_corpus.json
│   ├── HybridQA_dev.jsonl
│   ├── HybridQA_test.jsonl
│   └── HybridQA_train.jsonl
└── TATQA/
    ├── TATQA_corpus.json
    ├── TATQA_dev.jsonl
    ├── TATQA_test.jsonl
    └── TATQA_train.jsonl

baselines/
├── data/ # evaluation results for baselines for 3 datasets
│   ├── ConvFinQA/
│   ├── HybridQA/
│   └── TATQA/
├── scripts/  # scripts to run the baseline
└── src/  # baseline and evaluation code

benchmark_framework/
├── scripts/  # scripts for benchmark creation
└── src/      # code for benchmark creation
```
## How to run the baselines 


## How to run the benchmark creation 

## Benchmark statistics 
