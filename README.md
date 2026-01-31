<h1 align="center">A Systematic Cross-domain Evaluation of Document Retrievers</b></h1>

<h4 align="center">
    <p>
        <!--<a href="#links">Model/Data Links</a> |-->
        <!--<a href="#installation">Installation</a> |-->
        | <a href="#usage">Usage</a> |
        <a href="#available-models-and-datasets">Available Models and Datasets</a> |
        <!--<a href="https://huggingface.co/spaces/mteb/leaderboard?task=instructionretrieval">Leaderboard</a> |-->
        <a href="#citing">Citing</a> |
    <p>
</h4>

Official repository for the paper [A Systematic Cross-domain Evaluation of Document Retrievers]().


## Usage
To obtain the same results as in our paper, you need to:

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

2. Run the Python scripts in `run_scripts`. You can use `run_scripts/run_*.py` scripts to evaluate each of the models on the datasets. Inside each script, at the top, you will find all available models and datasets that you can use in this particular script. You can either run the scripts normally:

    ```bash
    python run_*.py
    ```

    or in the background:

    ```bash
    nohup python -u run_*.py > logs/run_*.log 2>&1 &
    ```

    where in both cases * is one of [base, gritlm, llm2vec, new, nvembed, repllama].


## Available Models and Datasets
### Models:

| Model | Hugging Face Checkpoint | Paper |
| ----- | ----------------------- | ----- |
| BM25 | | [Robertson et al., 1994]() [Robertson & Zaragoza, 2009]() |
| docT5query<sup>*</sup> | [doc2query/msmarco-t5-base-v1](https://huggingface.co/doc2query/msmarco-t5-base-v1) | [Nogueira & Lin, 2019]() |
| query2doc<sup>*</sup> | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Wang et al., 2023]() |
| DeepCT | [macavaney/deepct](https://huggingface.co/macavaney/deepct) | [Dai & Callan, 2020]() |
| SPARTA | [BeIR/sparta-msmarco-distilbert-base-v1](https://huggingface.co/BeIR/sparta-msmarco-distilbert-base-v1) | [Zhao et al., 2021]() |
| uniCOIL | [castorini/unicoil-msmarco-passage](https://huggingface.co/castorini/unicoil-msmarco-passage),<br> [castorini/unicoil-noexp-msmarco-passage](https://huggingface.co/castorini/unicoil-noexp-msmarco-passage) | [Lin & Ma, 2021]() |
| SPLADE-v3 | [naver/splade-v3](https://huggingface.co/naver/splade-v3) | [Lassance et al., 2024]() |
| DPR | [facebook/dpr-ctx_encoder-multiset-base](https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base),<br> [facebook/dpr-question_encoder-multiset-base](https://huggingface.co/facebook/dpr-question_encoder-multiset-base) | [Karpukhin et al., 2020]() |
| ColBERTv2 | [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | [Santhanam et al., 2022]() |
| TAS-B | [sentence-transformers/msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | [Hofstätter et al., 2021]() |
| ANCE | [castorini/ance-msmarco-passage](https://huggingface.co/castorini/ance-msmarco-passage) | [Xiong et al., 2021]() |
| SimCSE | [princeton-nlp/sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased) | [Gao et al., 2021]() |
| RetroMAE | [Shitao/RetroMAE_BEIR](https://huggingface.co/Shitao/RetroMAE_BEIR) | [Xiao et al., 2022]() |
| coCondenser | [Luyu/co-condenser-marco-retriever](https://huggingface.co/Luyu/co-condenser-marco-retriever) | [Gao & Callan, 2022]() |
| GTR (large) | [sentence-transformers/gtr-t5-large](https://huggingface.co/sentence-transformers/gtr-t5-large) | [Ni et al., 2022]() |
| Contriever | [facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco) | [Izacard et al., 2022]() |
| COCO-DR (large) | [OpenMatch/cocodr-large-msmarco](https://huggingface.co/OpenMatch/cocodr-large-msmarco) | [Yu et al., 2022]() |
| SimLM | [intfloat/simlm-base-msmarco-finetuned](https://huggingface.co/intfloat/simlm-base-msmarco-finetuned) | [Wang et al., 2023a]() |
| DRAGON+ | [facebook/dragon-plus-context-encoder](https://huggingface.co/facebook/dragon-plus-context-encoder),<br> [facebook/dragon-plus-query-encoder](https://huggingface.co/facebook/dragon-plus-query-encoder) | [Li et al., 2023a]() |
| HyDE<sup>*</sup> | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Gao et al., 2023]() |
| INSTRUCTOR (large) | [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large) | [Su et al., 2023]() |
| LLM2Vec<sup>&dagger;</sup> | [McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp)  | [BehnamGhader et al., 2024]() |
| RepLLaMA | [castorini/repllama-v1-7b-lora-passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) | [Ma et al., 2024]() |
| mE5 (large) | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | [Wang et al., 2024]() |
| GTE (large v1.5) | [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)  | [Li et al., 2023]() |
| BGE (large v1.5) | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | [Xiao et al., 2024]() |
| Jina Embeddings v3 | [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | [Sturua et al., 2024]() |
| Nomic Embed (v1.5) | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | [Nussbaum et al., 2025]() |
| GritLM | [GritLM/GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) | [Muennighoff et al., 2025]() |
| Qwen3 Embeddings (0.6B) | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)          | [Zhang et al., 2025]() |
| NV-Embed-v2 | [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) | [Lee et al., 2025]() |
| Embedding-Gemma | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | [Vera et al., 2025]() |
| KaLM-Embedding (v2.5) | [KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5](https://huggingface.co/KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5)| [Zhao et al., 2025]() |

Note: <sup>*</sup>Checkpoint of query/document generation model; <sup>&dagger;</sup>Commit **9d1613c**42e2f90050dc11daeb1a24919811fa2c5

### Datasets:

...


## Citing

```bibtex
@misc{velev2026retrievers,
      title={{A Systematic Cross-domain Evaluation of Document Retrievers}}, 
      author={Valentin Velev and Andreas Spitz},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```