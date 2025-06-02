import os
import json
import torch
import collections
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re

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


# 1. Load the Mintaka dataset (English test split)
dataset = load_dataset("AmazonScience/mintaka", split="test", name="en")

# 2. Load the LLaMA 3 model and tokenizer (Unsloth)
model_id = "unsloth/llama-3-8b-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print ("special tokens, eos_token:")
print(tokenizer.special_tokens_map)
print(tokenizer.eos_token)


# 3. Exact match function
def compute_exact_match(predictions, references):
    return {
        "exact_match": 100.0 * sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)
    }

# 4. Inference and evaluation loop
preds, refs = [], []

predictions = {}
em_scores = []
f1_scores = []

for example in tqdm(dataset):
    question = example["question"]
    reference = example["answerText"] if isinstance(example["answerText"], str) else example["answerText"][0]

    # Manual chat-style prompt
    #prompt = (
    #"<|begin_of_text|>"
    #"You are a knowledgeable assistant. "
    #"Answer the question with a short, simple response. Avoid explanations.\n\n"
    #f"Question:\n\n{question}\n\nAnswer:"
    #)

    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    messages = [
    {"role": "system", "content": "You are a knowledgeable assistant. Answer the question with a short, simple response. Avoid explanations."},
    {"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #print (prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract the assistant's answer
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = generated_text.strip()

    # Extract answer after <|assistant|>
    #if "Answer:" in generated_text:
    #    generated_answer = generated_text.split("Answer:")[-1].strip()
    #else:
    #    generated_answer = generated_text.strip()

    print ("\n\n\n")
    print ("question", question)
    print ("reference", reference)
    print ("generated_answer", generated_answer)


    preds.append(generated_answer)
    refs.append(reference.strip())
    em_scores.append(calculate_em(generated_answer, reference, mode="text"))
    f1_scores.append(calculate_f1(generated_answer, reference, mode="text"))
    print ("em_scores,f1_scores", em_scores[len(em_scores)-1],f1_scores[len(em_scores)-1])

print(f"Exact Match (EM): {100 * sum(em_scores) / len(em_scores):.2f}%")
print(f"F1 Score: {100 * sum(f1_scores) / len(f1_scores):.2f}%")
