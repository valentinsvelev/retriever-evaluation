nvembed_instructions = {
    "ClimateFEVER":
        {
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim",
            "corpus": ""
        },
    "HotpotQA":
        {
            "query": "Given a multi-hop question, retrieve documents that can help answer the question",
            "corpus": ""
        },
    "FEVER":
        {
            "query": "Given a claim, retrieve documents that support or refute the claim",
            "corpus": ""
        },
    "MSMARCO":
        {
            "query": "Given a web search query, retrieve relevant passages that answer the query",
            "corpus": ""
        },
    "DBPedia":
        {
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia",
            "corpus": ""
        },
    "NQ":
        {
            "query": "Given a question, retrieve passages that answer the question",
            "corpus": ""
        },
    "QuoraRetrieval":
        {
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question",
            "corpus": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
    "SCIDOCS":
        {
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
            "corpus": ""
        },
    "TRECCOVID":
        {
            "query": "Given a query on COVID-19, retrieve documents that answer the query",
            "corpus": ""
        },
    "Touche2020":
        {
            "query": "Given a question, retrieve passages that answer the question",
            "corpus": ""
        },
    "SciFact":
        {
            "query": "Given a scientific claim, retrieve documents that support or refute the claim",
            "corpus": ""
        },
    "NFCorpus":
        {
            "query": "Given a question, retrieve relevant documents that answer the question",
            "corpus": ""
        },
    "ArguAna":
        {
            "query": "Given a claim, retrieve documents that support or refute the claim",
            "corpus": ""
        },
    "FiQA2018":
        {
            "query": "Given a financial question, retrieve relevant passages that answer the query",
            "corpus": ""
        },
}

gritlm_instructions = {
    'ArguAna': {
        'query': 'Given a claim, find documents that refute the claim',
        'corpus': '',
    },
    'ClimateFEVER': {
        'query': 'Given a claim about climate change, retrieve documents that support or refute the claim',
        'corpus': '',
    },
    'CQADupstackRetrieval': {
        'query': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question',
        'corpus': '',
    },
    'DBPedia': {
        'query': 'Given a query, retrieve relevant entity descriptions from DBPedia',
        'corpus': '',
    },
    'FEVER': {
        'query': 'Given a claim, retrieve documents that support or refute the claim',
        'corpus': '',
    },
    'FiQA2018': {
        'query': 'Given a financial question, retrieve user replies that best answer the question',
        'corpus': '',
    },
    'HotpotQA': {
        'query': 'Given a multi-hop question, retrieve documents that can help answer the question',
        'corpus': '',
    },
    'MSMARCO': {
        'query': 'Given a web search query, retrieve relevant passages that answer the query',
        'corpus': '',
    },
    'NFCorpus': {
        'query': 'Given a question, retrieve relevant documents that best answer the question',
        'corpus': '',
    },
    'NQ': {
        'query': 'Given a question, retrieve Wikipedia passages that answer the question',
        'corpus': '',
    },
    'QuoraRetrieval': {
        'query': 'Given a question, retrieve questions that are semantically equivalent to the given question',
        'corpus': '',
    },
    'SCIDOCS': {
        'query': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
        'corpus': '',
    },
    'SciFact': {
        'query': 'Given a scientific claim, retrieve documents that support or refute the claim',
        'corpus': '',
    },
    'Touche2020': {
        'query': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
        'corpus': '',
    },
    'TRECCOVID': {
        'query': 'Given a query on COVID-19, retrieve documents that answer the query',
        'corpus': '',
    },
}

instructor_instructions = {
    'ClimateFEVER':
        {
            'query': 'Represent the Climate question for retrieving supporting documents: ',
            'corpus': 'Represent the document for retrieval: ',
        },
    'HotpotQA':
        {
            'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
            'corpus': 'Represent the document for retrieval: ',
        },
    'FEVER':
        {
            'query': 'Represent the fact for retrieving supporting evidence: ',
            'corpus': 'Represent the evidence for retrieval: ',
        },
    'MSMARCO':
        {
            'query': 'Represent the question for retrieving supporting documents: ',
            'corpus': 'Represent the document for retrieval: ',
        },
    'DBPedia':
        {
            'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
            'corpus': 'Represent the document for retrieval: ',
        },
    'NQ':
        {
            'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
            'corpus': 'Represent the document for retrieval: ',
        },
    'QuoraRetrieval':
        {
            'query': 'Represent the Quora question for retrieving duplicate questions: ',
            'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
        },
    'SCIDOCS':
        {
            'query': 'Represent a Science question for retrieving supporting papers: ',
            'corpus': 'Represent the Science paper: ',
        },
    'TRECCOVID':
        {
            'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
            'corpus': 'Represent the Coronavirus document for retrieval: ',
        },
    'Touche2020':
        {
            'query': 'Represent a question: ',
            'corpus': 'Represent an argument: ',
        },
    'SciFact':
        {
            'query': 'Represent a Scientific query for retrieving a supporting passage; ',
            'corpus': 'represent the Scientific passage for retrieval; ',
        },
    'NFCorpus':
        {
            'query': 'Represent the Medicine question for retrieving a relevant document: ',
            'corpus': 'Represent the medical document for retrieval: ',
        },
    'ArguAna':
        {
            'query': 'Represent a Debate argument for retrieving a counter-argument: ',
            'corpus': 'Represent a Counter-argument: ',
        },
    'CQADupstackTexRetrieval':
        {
            'query': 'Represent the question for retrieving answers: ',
            'corpus': 'Represent the answer for retrieval: ',
        },
    'CQADupstackWebmastersRetrieval':
        {
            'query': 'Represent the Webmaster question for retrieving answers: ',
            'corpus': 'Represent the Webmaster answer: ',
        },
    'CQADupstackEnglishRetrieval':
        {
            'query': 'Represent the English question for retrieving documents: ',
            'corpus': 'Represent the English answer for retrieval: ',
        },
    'CQADupstackGamingRetrieval':
        {
            'query': 'Represent the Gaming question for retrieving answers: ',
            'corpus': 'Represent the Gaming answer for retrieval: ',
        },
    'CQADupstackGisRetrieval':
        {
            'query': 'Represent the Gis question for retrieving answers: ',
            'corpus': 'Represent the Gis answer for retrieval: ',
        },
    'CQADupstackUnixRetrieval':
        {
            'query': 'Represent the Unix question for retrieving answers: ',
            'corpus': 'Represent the Unix answer for retrieval: ',
        },
    'CQADupstackMathematicaRetrieval':
        {
            'query': 'Represent the Mathematical question for retrieving answers: ',
            'corpus': 'Represent the Mathematical answer for retrieval: ',
        },
    'CQADupstackStatsRetrieval':
        {
            'query': 'Represent the Statistical question for retrieving answers: ',
            'corpus': 'Represent the Statistical answer for retrieval: ',
        },
    'CQADupstackPhysicsRetrieval':
        {
            'query': 'Represent the Physics question for retrieving answers: ',
            'corpus': 'Represent the Physics answer for retrieval: ',
        },
    'CQADupstackProgrammersRetrieval':
        {
            'query': 'Represent the Programming question for retrieving answers: ',
            'corpus': 'Represent the Programming answer for retrieval: ',
        },
    'CQADupstackAndroidRetrieval':
        {
            'query': 'Represent the Android question for retrieving answers: ',
            'corpus': 'Represent the Android answer for retrieval: ',
        },
    'CQADupstackWordpressRetrieval':
        {
            'query': 'Represent the Wordpress question for retrieving answers: ',
            'corpus': 'Represent the Wordpress answer for retrieval: ',
        },
    'FiQA2018':
        {
            'query': 'Represent the finance question for retrieving the supporting answers: ',
            'corpus': 'Represent the finance answer for retrieval: ',
        },
}


kalm_instructions = {
    "QuoraRetrieval": "Instruct: Retrieve semantically similar questions \n Query: ",
    "CQADupstack": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the givenquestion \n Query: "
}

llm2vec_instructions = {
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim:",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim:",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query:",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia:",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question:",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question:",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper:",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query:",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question:",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question:",
    "ArguAna": "Given a claim, find documents that refute the claim:",
    "CQADupstackTexRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWebmastersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackEnglishRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGisRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackMathematicaRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackStatsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackPhysicsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackProgrammersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackAndroidRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWordpressRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question:",
}