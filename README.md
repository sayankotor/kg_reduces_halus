0. For load Mistral-7B checkpoints:

**cd kg_reduces_halu**

**wget 'https://www.dropbox.com/scl/fi/yk35riqrumf90n3po5azz/chkpts1.tar.gz?rlkey=duylp86l0kju23xggr7zb2cu5&st=3826jrv9&dl=1' -O chkpts.tar.gz**

**tar -xvzf chkpts.tar.gz**


This is a code for evaluations, which supports the work "Addressing Hallucinations in Language Models Knowledge Graph Embeddings as an Additional Modality"


1. The evaluation on the Halu eval benchmark is based on the original repository (https://github.com/RUCAIBox/HaluEval) with minor changes.
For evaluation, run

**cd HaluEval**

**python evaluate.py --task task --model model**

Available tasks are: qa, summarization, dialogue
In this anonimized repository models mistral7b, mistral7b-kg are availible.

2. For True-False evaluation:

**cd True-False**

**python evaluate.py --model model --kg**

3. In the folder 'dataset_example' there is a part of the dataset described in the paper.
