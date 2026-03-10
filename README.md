1. Comparing two answer: embedding model

        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")

        s1 = "The cat is sleeping on the sofa"
        s2 = "A cat is lying on the couch"

        emb = model.encode([s1, s2])

        similarity = cosine_similarity([emb[0]], [emb[1]])[0][0]
        print(similarity)




TODO list:
support tool calling workflow
support TTRL training
add image crop tools        