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

## Benchmark statistics 

#### Clusters, DPRs, validation and filtering, final DPRs, and the corpus sizes

| Dataset   | Split |   # of Clusters |   # of Generated DPRs |   DPR quality issues |   DPR-DP Alignment Issues |   Empty GT after filtering |   # of filtered DPRs |   # of Tables |   # of Text Passages |
|-----------|:------|----------------:|----------------------:|---------------------:|--------------------------:|---------------------------:|---------------------:|--------------:|---------------------:|
| Hybrid QA | Train |            1111 |                  5555 |                   12 |                       654 |                        46 |                 4843 |         12378 |               41,608 |
|           | Dev   |             427 |                  2135 |                   13 |                       103 |                        11 |                 2008 |                ↑|                      ↑|
|           | Test  |             433 |                  2165 |                   79 |                       100 |                         6 |                 1980 |                ↑|                      ↑|
| TATQA     | Train |             314 |                  1570 |                    7 |                       406 |                       337 |                  820 |          2757 |                4,760 |
|           | Dev   |              55 |                   275 |                    5 |                        59 |                        64 |                  147 |                ↑|                      ↑|
|           | Test  |              58 |                   290 |                    5 |                        40 |                        69 |                  176 |                ↑|                      ↑|
| ConvFinQA | Train |             564 |                  2820 |                    6 |                       316 |                       385 |                 2113 |          4976 |                8721 |
|           | Dev   |              97 |                   485 |                    5 |                        54 |                        53 |                  373 |                ↑|                      ↑|
|           | Test  |             206 |                  1030 |                    6 |                       149 |                       248 |                  627 |                ↑|                      ↑|

#### Groud truth tables and text distributions

| Dataset     | Split | TBL Mean |TBL Min | TBL Max |  TBL STD | TXT Mean | TXT Min | TXT Max |  TXT STD |
|-------------|-------|-----------------:|----:|----:|-----:|---------------:|----:|----:|-----:|
| **Hybrid QA** | Train |             8.56 |   1 |  72 |  7.32 |          28.74 |   2 | 240 |  7.32 |
|             | Dev   |             5.24 |   1 |  31 |  3.64 |          17.70 |   2 | 105 | 12.21 |
|             | Test  |             5.04 |   1 |  25 |  3.30 |          16.93 |   2 |  78 | 11.24 |
| **TATQA**     | Train |             4.01 |   1 |  34 |  3.27 |           6.98 |   1 |  65 |  6.26 |
|             | Dev   |             3.58 |   1 |   9 |  1.73 |           6.04 |   1 |  14 |  3.02 |
|             | Test  |             3.35 |   1 |   9 |  1.53 |           5.69 |   1 |  17 |  3.01 |
| **ConvFinQA** | Train |             4.33 |   1 |  17 |  2.29 |           7.59 |   1 |  38 |  4.84 |
|             | Dev   |             3.53 |   1 |   8 |  1.55 |           6.40 |   1 |  18 |  3.41 |
|             | Test  |             6.88 |   2 |  17 |  3.58 |          12.37 |   2 |  48 |  9.02 |




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

The goal of this step is to cluster the questions (which are already grouped in to tables) by identifying tables that share common analytical themes. Questions from the same table are concatenated to form a document corresponds that table and those documents are clustered by topic using `BERTopic`. Once the clustering is done, cluster quality metrics are calculated on the resulting clusters.


```commandline
sh benchmark_framework/scripts/1_cluster.sh
```

It will generate the following files. 
- `*-clusters.json` files contain the clusters for each dataset/split with the clusters that contain several tables grouped by the topic. 
- `*-clusters_summary.json` files contain cluster quality metrics calculated at each individual cluster level and globably. Global metrics include  silhouette_score, calinski_harabasz_index, davies_bouldin_index and individual cluster metrics include silhouette, intra_cluster_mse, inter_cluster_dist, and db_component. Statistics such as how many tables, questions in the cluster also recorded in tis file. 
```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   ├── ConvFinQA_dev_clusters.json
│   │   └── ConvFinQA_dev_clusters_summary.json
│   ├── test
│   │   ├── ConvFinQA_test_clusters.json
│   │   └── ConvFinQA_test_clusters_summary.json
│   └── train
│       ├── ConvFinQA_train_clusters.json
│       └── ConvFinQA_train_clusters_summary.json
├── HybridQA
│   ├── dev
│   │   ├── HybridQA_dev_clusters.json
│   │   └── HybridQA_dev_clusters_summary.json
│   ├── test
│   │   ├── HybridQA_test_clusters.json
│   │   └── HybridQA_test_clusters_summary.json
│   └── train
│       ├── HybridQA_train_clusters.json
│       └── HybridQA_train_clusters_summary.json
└── TATQA
    ├── dev
    │   ├── TATQA_dev_clusters.json
    │   └── TATQA_dev_clusters_summary.json
    ├── test
    │   ├── TATQA_test_clusters.json
    │   └── TATQA_test_clusters_summary.json
    └── train
        ├── TATQA_train_clusters.json
        └── TATQA_train_clusters_summary.json
```

#### Data Product Filtering

Even though topic clusters can provide some grouping of table that share a common topic, they might not be in the same level of granualarity of a data product Real-world data products need tighter semantic coherence and a manageable scope to be practically useful. This step refines raw clusters by incorporating table schema information and controlling granularity. This step will also ensure that the size of the data products are  manageable.

```commandline
sh benchmark_framework/scripts/2_filtering.sh
```

It will generate the following files with the refined clusters.
```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   └── ConvFinQA_dev_filtered_clusters.json
│   ├── test
│   │   └── ConvFinQA_test_filtered_clusters.json
│   └── train
│       └── ConvFinQA_train_filtered_clusters.json
├── HybridQA
│   ├── dev
│   │   └── HybridQA_dev_filtered_clusters.json
│   ├── test
│   │   └── HybridQA_test_filtered_clusters.json
│   └── train
│       └── HybridQA_train_filtered_clusters.json
└── TATQA
    ├── dev
    │   └── TATQA_dev_filtered_clusters.json
    ├── test
    │   └── TATQA_test_filtered_clusters.json
    └── train
        └── TATQA_train_filtered_clusters.json
```

#### Data Product Request Generation

> **Note:**  From this step forward, the framework will be using LLM calls for various tasks. It uses [DSPy](https://dspy.ai/) for LLM calls which based on [LiteLLM](https://www.litellm.ai/). LiteLLM supports a large number of LLM provides including commercial providers as well as Ollama, vLLM, etc. as decribed [here](https://docs.litellm.ai/docs/providers). Please configure the [llm_provider.py](benchmark_framework/src/llm_provider.py) to based on the LLM provider you have access to.

In this step, we generate Data Product Requests for each of the filtered clusters. 

```commandline
sh benchmark_framework/scripts/3_generation.sh
```

#### Data Product Request Validation

```commandline
sh benchmark_framework/scripts/4_eval.sh
```


#### Data Product Ground Truth Refinement

```commandline
sh benchmark_framework/scripts/5_refinement.sh
```


