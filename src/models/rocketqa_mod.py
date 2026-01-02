import rocketqa

print(rocketqa.available_models())

dual = rocketqa.load_model(
    model="v2_marco_de",
    use_cuda=True,
    #device_id=0,
    batch_size=16
)

query_list = ["trigeminal definition"]
para_list = [
    "Definition of TRIGEMINAL: of or relating to the trigeminal nerve."
]

# Vector embeddings
q_embs = dual.encode_query(query=query_list)
p_embs = dual.encode_para(para=para_list)

# Similarity scores (dot-product)
scores = dual.matching(query=query_list, para=para_list)
print(scores)