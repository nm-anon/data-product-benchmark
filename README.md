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

> **Note:** HybridQA is the largest of the datasets, and running this baseline may be slow (running locally with no GPU, producing the entire collection of text embeddings may take about an hour). Embedding speed will be much faster if you are running on a machine with GPU support. Producing baseline results for TATQA and ConfFinQA is expected to finish within a few minutes.

## How to run the benchmark creation 

> **Note:**  Systems that plan to use benchmark can use the benchmark directly from the data as shown in the running baseline section, and no need to re-run the benchmark creation process. Benchmark creation is documented here for reproducibility.

#### Downloading the existing QA Benchmarks
The benchmark uses data from the following existing repositories and you will need to download those repositories first. 
- [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks.git)
- [HybridQA](https://github.com/wenhuchen/HybridQA.git)
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA.git)
- [ConvFinQA](https://github.com/czyssrs/ConvFinQA.git)

To make the process easier, we have added them as git-submodules. Use the following command to clone all of them in a single command.

```commandline
git submodule update --init --recursive
```

The ConvFinQA dataset is in a compressed zip file. Unzip it using the following command. 
```commandline
unzip benchmark_framework/data/raw/ConvFinQA/data.zip -d benchmark_framework/data/raw/ConvFinQA
```

#### Corpus preparation

The following command will run the corpus preparation for HybridQA, TATQA, and ConvFinQA datasets. It reads the raw data from the original Git repos files and create tables and text corpora in a common format that will be used by the next phases of the pipeline.

```commandline
sh benchmark_framework/scripts/0_prepare.sh
```

This step will create the following files.

```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   └── ConvFinQA_dev_corpus.json
│   ├── test
│   │   └── ConvFinQA_test_corpus.json
│   └── train
│       └── ConvFinQA_train_corpus.json
├── HybridQA
│   ├── dev
│   │   └── HybridQA_dev_corpus.json
│   ├── test
│   │   └── HybridQA_test_corpus.json
│   └── train
│       └── HybridQA_train_corpus.json
└── TATQA
    ├── dev
    │   └── TATQA_dev_corpus.json
    ├── test
    │   └── TATQA_test_corpus.json
    └── train
        └── TATQA_train_corpus.json
```

#### Topic clustering of questions (grouped by the tables)


```commandline
sh benchmark_framework/scripts/0_prepare.sh
```

## Benchmark statistics 

