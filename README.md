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
debug on mmsearch without tools
0.11 -> 0.25
add tools and reasoning chain

3. add image crop tools and code intepreter


conda create -n mmsearch_r1 python==3.10 -y
conda activate mmsearch_r1
pip3 install -e verl_grpo/verl
pip3 install vllm==0.8.2
pip3 install transformers==4.51.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip install ray==2.43.0