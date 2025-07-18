import os
import torch
import collections
import re
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda:1"

# 1. Load the Mintaka dataset
dataset = load_dataset("AmazonScience/mintaka", split="test", name="en")

# 2. Load the model and tokenizer for Llama 3.2 1B
model_id = "unsloth/llama-3.2-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": DEVICE}
)
model.eval()

#new_eos_token = "<|eos|>"

# Set it and add to tokenizer
#tokenizer.eos_token = new_eos_token
#tokenizer.add_special_tokens({'eos_token': new_eos_token})

# Check ID
print("New EOS token ID:", tokenizer.eos_token_id)

special_tokens = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "pad_token": tokenizer.pad_token,
    "unk_token": tokenizer.unk_token,
    "additional_special_tokens": tokenizer.additional_special_tokens,
}

special_token_ids = {
    name: tokenizer.convert_tokens_to_ids(token)
    for name, token in special_tokens.items()
    if isinstance(token, str) or token is not None
}

print("Special tokens:", special_tokens)
print("Token IDs:", special_token_ids)

# 3. Define helper functions
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
em_scores = []
f1_scores = []

for example in tqdm(dataset):
    question = example["question"]
    reference = example["answerText"] if isinstance(example["answerText"], str) else example["answerText"][0]

    # Format using chat template
    prompt = f"You are a knowledgeable assistant. Answer the question with a short, simple response. Avoid explanations.\n\n{question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)

    outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = generated_text.strip()

    print("\n\n\n")
    print("question", question)
    print("reference", reference)
    print("generated_answer", generated_answer)

    em_scores.append(calculate_em(generated_answer, reference, mode="text"))
    f1_scores.append(calculate_f1(generated_answer, reference, mode="text"))
    print("em_scores,f1_scores", em_scores[-1], f1_scores[-1])

# Final scores
print(f"Exact Match (EM): {100 * sum(em_scores) / len(em_scores):.2f}%")
print(f"F1 Score: {100 * sum(f1_scores) / len(f1_scores):.2f}%")
