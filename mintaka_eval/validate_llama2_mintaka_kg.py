import os
import json
import torch
import collections
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re

DEVICE = "cuda:1"
# 1. Load the Mintaka dataset
dataset = load_dataset("AmazonScience/mintaka", split="test", name="en")

# 2. Load the model and tokenizer (no accelerate issues)
model_id = "unsloth/llama-2-7b"  # or "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": DEVICE}
)
model.eval()
            
projection = torch.load("/home/jovyan/shares/SR004.nfs2/chekalina/kg_reduces_halus/notebook_new/ckpts/projection_llama2_renorm", map_location=DEVICE)
start_emb = torch.load("/home/jovyan/shares/SR004.nfs2/chekalina/kg_reduces_halus/notebook_new/ckpts/SOI_llama2_renorm.pt", map_location=DEVICE)
end_emb = torch.load("/home/jovyan/shares/SR004.nfs2/chekalina/kg_reduces_halus/notebook_new/ckpts/EOI_llama2_renorm.pt", map_location=DEVICE)

print("loads", flush = True)

projection = projection.to(dtype=model.dtype, device=DEVICE)
start_emb = start_emb.to(dtype=model.dtype, device=DEVICE)
end_emb = end_emb.to(dtype=model.dtype, device=DEVICE)

import pickle

with open('/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/big_graph/name2num.pickle', 'rb') as handle:
    name2num = pickle.load(handle)
    
print("len", len(name2num), flush = True)

with open('/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/big_graph/dicts/embedds.pickle', 'rb') as handle:
    embedds = pickle.load(handle)
    
print("len embedds", len(embedds),flush = True)

def get_embedding(ent):
    num = name2num.get(ent, -111)
    emb = embedds.get(num, -111)
    return emb

# 3. Evaluation metric
def normalize_and_tokenize_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower().split()

def calculate_em(pred: Union[list, str, None], answer: Union[list, str, None], mode: str) -> int:
    if mode == 'text' and pred and answer:
        pred = normalize_and_tokenize_text(pred)
        answer = normalize_and_tokenize_text(answer)
        for i in range(0, len(pred) - len(answer) + 1):
            if answer == pred[i: i + len(answer)]:
                return 1
        return 0
    else:
        return int(pred == answer)

def calculate_f1(pred: Union[str, List], answer: Union[str, List], mode: str) -> float:
    if not answer or not pred:
        return int(answer == pred)
    if mode == 'text':
        pred = pred.split()
        answer = answer.split()
    common = collections.Counter(pred) & collections.Counter(answer)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(answer)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# 4. Inference loop
predictions = {}
em_scores = []
f1_scores = []

for example in tqdm(dataset):
    question = example["question"]
    reference = example["answerText"] if isinstance(example["answerText"], str) else example["answerText"][0]
    ents= [elem['name'] for elem in example['questionEntity']]

    #prompt = f"You are a helpful assistant. Answer the following question briefly and clearly. Do not repeat or provide additional information except the answer. Do NOT provide an explanation. Only provide the answer.\n\n{question}\n\nAnswer:"

    prompt = f"You are a knowledgeable assistant. Answer the question with a short, simple response. Avoid explanations.\n\n{question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)

    ent_embs = [get_embedding(ent) for ent in ents]
    ent_embs = [elem for elem in ent_embs if elem != -111]

    
    if (len(ent_embs)>0):
        try:
            ent_embs = torch.tensor(ent_embs).to(device=DEVICE, dtype=model.dtype)
            m = ent_embs.mean(1, keepdim=True)
            s = ent_embs.std(1, unbiased=False, keepdim=True)
            ent_embs -= m
            ent_embs /= s
        except:
            print ("exception", len(ent_embs), ent_embs)
        projected_kg_embeddings = projection(ent_embs[None,:,:])
        embeddings = torch.cat(
                   [
                        inputs_embeds,
                        start_emb[None, None, ...],
                        projected_kg_embeddings,
                        end_emb[None, None, ...],
                    ],
                dim=1,
            ).to(dtype=torch.float16, device=DEVICE)
    else:
        embeddings = inputs_embeds.to(dtype=torch.float16, device=DEVICE)
        
    outputs = model.generate(inputs_embeds=embeddings, max_new_tokens = 50)
    #out = out[:, 1:]
    #generated_texts = tokenizer.batch_decode(out)[0]
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    generated_answer = generated_text.strip()
    print ("\n\n\n")
    print ("question", question)
    print ("reference", reference)
    print ("generated_answer", generated_answer)

    em_scores.append(calculate_em(generated_answer, reference, mode="text"))
    f1_scores.append(calculate_f1(generated_answer, reference, mode="text"))
    print ("em_scores,f1_scores", em_scores[len(em_scores)-1],f1_scores[len(em_scores)-1])

print(f"Exact Match (EM): {100 * sum(em_scores) / len(em_scores):.2f}%")
print(f"F1 Score: {100 * sum(f1_scores) / len(f1_scores):.2f}%")