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

# Load dataset
dataset = load_dataset("AmazonScience/mintaka", split="test", name="en")

# Load model and tokenizer
model_id = "unsloth/mistral-7b-instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

unk_id = tokenizer.encode("<unk>", add_special_tokens=False)[0]
tokenizer.pad_token_id = 2
tokenizer.eos_token_id = 0

model.resize_token_embeddings(len(tokenizer))
N_EMBEDDINGS = model.model.embed_tokens.weight.shape[0]
print("Number of embeddings in tokenizer:", N_EMBEDDINGS)

model.eval()

# Output setup
output_dir = "./mintaka_predictions"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "predictions.json")

predictions = {}
em_scores = []
f1_scores = []

# Inference loop
for example in tqdm(dataset):
    example_id = example["id"]
    question = example["question"]
    reference = example["answerText"]
    if not isinstance(reference, str):
        reference = reference[0]

    messages = [
    {"role": "system", "content": "You are a knowledgeable assistant. Answer the question with a short, simple response. Avoid explanations."},
    {"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #print (prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #if "Answer:" in output_text:
    #    generated_answer = output_text.split("Answer:")[-1].strip()
    #else:
    generated_answer = output_text.strip()

    print ("\n\n\n")
    print ("question", question)
    print ("reference", reference)
    print ("\n")
    print ("answer", generated_answer)

    predictions[example_id] = generated_answer
    em_scores.append(calculate_em(generated_answer, reference, mode="text"))
    f1_scores.append(calculate_f1(generated_answer, reference, mode="text"))



# Evaluation results
print(f"Exact Match (EM): {100 * sum(em_scores) / len(em_scores):.2f}%")
print(f"F1 Score: {100 * sum(f1_scores) / len(f1_scores):.2f}%")
