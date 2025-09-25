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

### Initial setup

Create a python environment and install requirements. Reproducibility has been checked for `python version 3.12`

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH="./"
```

To download data and produce baseline results for HybridQA, you will need to use git lsf due to some large file sizes.

If you have never installed git lfs before, follow the instructions at https://git-lfs.com/

Then activate git lfs and pull relevant data
```commandline
git lfs install
git lfs pull
```

### Run baseline retrieval experiments

Baseline experiments can be directly run using the following script:

`./baselines/scripts/run_baseline.sh`

You may also need to first make the sh file executable, e.g. `chmod +x baselines/scripts/run_baseline.sh`

Running the baseline script will proceed with producing embeddings and running baseline retrieval methods for a single database at a time.
The choice of which data will be used to produce results, as well as the choice of embedding model, can be changed within the `run_baselines.sh` script -- see commented lines in the file for specific arg choices.

After running the baseline script, results will be output to files like `baselines/data/ConfFinQA/ConfFinQA_test_results_eval_granite.json`

HybridQA is the largest of the datasets, and running this baseline may be slow (running locally with no GPU, producing the entire collection of text embeddings may take about an hour).
Embedding speed will be much faster if you are running on a machine with GPU support.
Producing baseline results for TATQA and ConfFinQA is expected to finish within a few minutes.
## How to run the benchmark creation 

## Benchmark statistics 
