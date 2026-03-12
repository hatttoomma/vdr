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
support TTRL training:
1. implement VLM GRPO using verl
OOM fix: limit context length by resizing iamge
test: debug on realworldqa
debug on mmsearch without tools
2. modify reward to majority vote and modify training data to unlabled data
debug on realworldqa


3. add image crop tools and code intepreter