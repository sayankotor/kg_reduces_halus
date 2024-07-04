This is a code for evaluations, which supports the work "Addressing Hallucinations in Language Models Knowledge Graph Embeddings as an Additional Modality"


1. The evaluation on the Halu eval benchmark is based on the original repository with minor changes.
For evaluation, run

**cd HaluEval**
**python evaluate.py --task task --model model**

Available tasks are: qa, summarization, dialogue
Available models are: mistral7b, mistral7b-kg, llama27b, llama27b-kg, llama3, llama3-kg

2. For True-False evaluation:

**cd True-False**
**python evaluate.py --model model --kg **

3. In the folder 'dataset_example' there is a part of the dataset described in the paper.
