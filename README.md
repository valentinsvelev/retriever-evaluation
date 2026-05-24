<h1 align="center">A Systematic Multi-Domain Evaluation of Document Retrievers</b></h1>

<h4 align="center">
    <p>
        <!--<a href="#links">Model/Data Links</a> |-->
        <!--<a href="#installation">Installation</a> |-->
        | <a href="#structure">Structure</a> |
        <a href="#reproducing-the-results">Reproducing the Results</a> |
        <a href="#available-models">Available Models</a> |
        <a href="#available-datasets">Available Datasets</a> |
        <!--<a href="https://huggingface.co/spaces/mteb/leaderboard?task=instructionretrieval">Leaderboard</a> |-->
    <p>
</h4>

## Structure

This repository is organized as follows:

- `requirements/`: environment-specific dependency files.
- `run_scripts/`: scripts for running evaluations.
- `src/`: core implementation of the retrieval evaluation framework.
  - `analysis/`: scripts for analyzing outputs and generating tables.
  - `configs/`: dataset, model, and instruction configurations.
  - `encoders/`: dense and sparse encoder abstractions.
  - `indexing/`: FAISS and Lucene indexing implementations.
  - `models/`: model-specific wrappers.
  - `data_handler.py`: data loading and preprocessing.
  - `evaluator.py`: retrieval evaluation pipeline.
  - `misc.py`: shared utility functions.
  - `run.py`: main execution entry point.
- `analysis.ipynb`: notebook for interactive analysis and reproducing results in paper.
- `README.md`: repository overview and usage instructions.

## Reproducing the Results
To obtain the same experimental results as in our paper, you need to:

0. Create four different virtual environments (venv), one for the older models, one for the newer models, one for LLM2Vec, and one for NV-Embed-v2 using

    ```bash 
    python -m venv venv_*
    ```

    where * is one of [old, new, llm2vec, nvembed].

1. Install the required packages for each venv using pip:

    ```bash
    source venv_*/bin/activate
    pip install -r requirements_*.txt
    ```

    where * is one of [old, new, llm2vec, nvembed].

2. Run the Python scripts in `run_scripts`. You can use `run_scripts/run_*.py` scripts to evaluate each of the models on the datasets. The available dataset and model IDs are available in the file `available_models_and_datasets.txt`. You can also find them in `src/configs/models.py` You can either run the scripts normally:

    ```bash
    python run_*.py
    ```

    or in the background:

    ```bash
    nohup python -u run_*.py > logs/run_*.log 2>&1 &
    ```

    where in both cases * is one of [base, gritlm, llm2vec, new, nvembed, repllama].

    Note that for ColBERTv2, the file `src/models/colbert.py` should be run to reproduce the results.

To obtain the same analysis results (e.g., plots and tables), you need to use the Jupyter Notebook `analysis_nb.ipynb`.


## Available Models

| Model | Hugging Face Model Checkpoint(s) | Paper |
| ----- | -------------------------- | ----- |
| BM25 | | [Robertson & Zaragoza (2009)](https://dl.acm.org/doi/10.1561/1500000019) |
| docT5query<sup>*</sup> | [doc2query/msmarco-t5-base-v1](https://huggingface.co/doc2query/msmarco-t5-base-v1) | [Nogueira & Lin (2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf) |
| query2doc<sup>*</sup> | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Wang et al. (2023)](https://aclanthology.org/2023.emnlp-main.585/) |
| DeepCT | [macavaney/deepct](https://huggingface.co/macavaney/deepct) | [Dai & Callan (2020)](https://dl.acm.org/doi/10.1145/3397271.3401204) |
| SPARTA | [BeIR/sparta-msmarco-distilbert-base-v1](https://huggingface.co/BeIR/sparta-msmarco-distilbert-base-v1) | [Zhao et al. (2021)](https://aclanthology.org/2021.naacl-main.47/) |
| uniCOIL | [castorini/unicoil-msmarco-passage](https://huggingface.co/castorini/unicoil-msmarco-passage),<br> [castorini/unicoil-noexp-msmarco-passage](https://huggingface.co/castorini/unicoil-noexp-msmarco-passage) | [Lin & Ma (2021)](https://arxiv.org/abs/2106.14807) |
| SPLADE-v3 | [naver/splade-v3](https://huggingface.co/naver/splade-v3) | [Lassance et al. (2024)](https://arxiv.org/abs/2403.06789) |
| DPR | [facebook/dpr-ctx_encoder-multiset-base](https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base),<br> [facebook/dpr-question_encoder-multiset-base](https://huggingface.co/facebook/dpr-question_encoder-multiset-base) | [Karpukhin et al. (2020)](https://aclanthology.org/2020.emnlp-main.550/) |
| ColBERTv2 | [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | [Santhanam et al. (2022)](https://aclanthology.org/2022.naacl-main.272/) |
| TAS-B | [sentence-transformers/msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | [Hofstätter et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3404835.3462891) |
| ANCE | [castorini/ance-msmarco-passage](https://huggingface.co/castorini/ance-msmarco-passage) | [Xiong et al. (2021)](https://openreview.net/forum?id=zeFrfgyZln) |
| SimCSE | [princeton-nlp/sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased) | [Gao et al. (2021)](https://aclanthology.org/2021.emnlp-main.552/) |
| RetroMAE | [Shitao/RetroMAE_BEIR](https://huggingface.co/Shitao/RetroMAE_BEIR) | [Xiao et al. (2022)](https://aclanthology.org/2022.emnlp-main.35/) |
| coCondenser | [Luyu/co-condenser-marco-retriever](https://huggingface.co/Luyu/co-condenser-marco-retriever) | [Gao & Callan (2022)](https://aclanthology.org/2022.acl-long.203/) |
| GTR (large) | [sentence-transformers/gtr-t5-large](https://huggingface.co/sentence-transformers/gtr-t5-large) | [Ni et al. (2022)](https://aclanthology.org/2022.emnlp-main.669/) |
| Contriever | [facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco) | [Izacard et al. (2022)](https://openreview.net/forum?id=jKN1pXi7b0) |
| COCO-DR (large) | [OpenMatch/cocodr-large-msmarco](https://huggingface.co/OpenMatch/cocodr-large-msmarco) | [Yu et al. (2022)](https://aclanthology.org/2022.emnlp-main.95/) |
| SimLM | [intfloat/simlm-base-msmarco-finetuned](https://huggingface.co/intfloat/simlm-base-msmarco-finetuned) | [Wang et al. (2023)](https://aclanthology.org/2023.acl-long.125/) |
| DRAGON+ | [facebook/dragon-plus-context-encoder](https://huggingface.co/facebook/dragon-plus-context-encoder),<br> [facebook/dragon-plus-query-encoder](https://huggingface.co/facebook/dragon-plus-query-encoder) | [Li et al. (2023)](https://aclanthology.org/2023.findings-emnlp.423/) |
| HyDE<sup>*</sup> | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Gao et al. (2023)](https://aclanthology.org/2023.acl-long.99/) |
| INSTRUCTOR (large) | [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large) | [Su et al. (2023)](https://aclanthology.org/2023.findings-acl.71/) |
| LLM2Vec<sup>&dagger;</sup> | [McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp)  | [BehnamGhader et al. (2024)](https://openreview.net/forum?id=IW1PR7vEBf) |
| RepLLaMA | [castorini/repllama-v1-7b-lora-passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) | [Ma et al. (2024)](https://dl.acm.org/doi/10.1145/3626772.3657951) |
| mE5 (large) | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | [Wang et al. (2024)](https://arxiv.org/abs/2402.05672) |
| GTE (large v1.5) | [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)  | [Li et al. (2023)](https://arxiv.org/abs/2308.03281) |
| BGE (large v1.5) | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | [Xiao et al. (2024)](https://dl.acm.org/doi/10.1145/3626772.3657878) |
| Jina Embeddings v3 | [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | [Sturua et al. (2025)](https://link.springer.com/chapter/10.1007/978-3-031-88720-8_21) |
| Nomic Embed (v1.5) | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | [Nussbaum et al. (2025)](https://openreview.net/forum?id=IPmzyQSiQE) |
| GritLM | [GritLM/GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) | [Muennighoff et al. (2025)](https://openreview.net/forum?id=BC4lIvfSzv) |
| Qwen3 Embeddings (0.6B) | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [Zhang et al. (2025)](https://arxiv.org/abs/2506.05176) |
| NV-Embed-v2 | [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) | [Lee et al. (2025)](https://openreview.net/forum?id=lgsyLSsDRe) |
| Embedding-Gemma | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | [Vera et al. (2025)](https://arxiv.org/abs/2509.20354) |
| KaLM-Embedding (v2.5) | [KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5](https://huggingface.co/KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5)| [Zhao et al. (2026)](https://openreview.net/forum?id=Y7qzhvWhcz) |

Note: <sup>*</sup>Checkpoint of query/document generation model; <sup>&dagger;</sup>Commit **9d1613c**42e2f90050dc11daeb1a24919811fa2c5

## Available Datasets

<table>
  <tr>
    <th>Type</th>
    <th>Dataset</th>
    <th>Paper</th>
  </tr>
  <tr>
    <td rowspan="3">In-domain</td>
    <td>MS MARCO</td>
    <td><a href="https://arxiv.org/abs/1611.09268">Bajaj et al. (2018)</a></td>
  </tr>
  <tr>
    <td>TREC-DL 2019</td>
    <td><a href="https://arxiv.org/abs/2003.07820">Craswell et al. (2020)</a></td>
  </tr>
  <tr>
    <td>TREC-DL 2020</td>
    <td><a href="https://arxiv.org/abs/2102.07662">Craswell et al. (2021)</a></td>
  </tr>
  <tr>
    <td rowspan="2">Out-of-domain</td>
    <td>BEIR</td>
    <td><a href="https://openreview.net/forum?id=wCu6T5xFjeJ">Thakur et al. (2021)</a></td>
  </tr>
  <tr>
    <td>LoTTE</td>
    <td><a href="https://aclanthology.org/2022.naacl-main.272/">Santhanam et al. (2022)</a></td>
  </tr>
  <tr>
    <td rowspan="2">Instruction-following</td>
    <td>InstructIR</td>
    <td><a href="https://arxiv.org/abs/2402.14334">Oh et al. (2024)</a></td>
  </tr>
  <tr>
    <td>FollowIR</td>
    <td><a href="https://aclanthology.org/2025.naacl-long.597/">Weller et al. (2025)</a></td>
  </tr>
</table>
