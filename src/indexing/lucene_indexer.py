################################################################################
# Title
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import os
import subprocess


class LuceneIndexer:
    """
    Handles Lucene (Anserini) indexing and searching.
    """
    def __init__(self, model_name: str, dataset_name: str, index_dir: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.index_dir = index_dir
        self.type = None

    def build(self, corpus_dir: str):
        print(f"Creating index at '{self.index_dir}'...")
        
        if self.model_name in ["bm25", "doc2query"]:
            cmd = ["python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", corpus_dir,
                    "--index", self.index_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "8",
                    "--storePositions", "--storeDocvectors", "--storeRaw"]
            self.type = "base"

        elif self.model_name == "deepct":
            cmd = ["python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", corpus_dir,
                    "--index", self.index_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "8",
                    "--storePositions", "--storeDocvectors", "--storeRaw", "--optimize"]
            self.type = "base"

        else:
            cmd = ["python", "-m", "pyserini.index.lucene",
                    "--collection", "JsonVectorCollection",
                    "--input", corpus_dir,
                    "--index", self.index_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "8",
                    "--impact", 
                    "--pretokenized", 
                    "--optimize"]
            self.type = "impact"

        subprocess.run(cmd, check=True)
