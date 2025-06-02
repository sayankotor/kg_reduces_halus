import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# 1. Load the Mintaka dataset (English test split)
dataset = load_dataset("AmazonScience/mintaka", split="test", name="en")

# 2. Load the Meta-Llama-3.2-3B-Instruct model and tokenizer
model_id = "unsloth/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 3. Exact match scoring function
def compute_exact_match(predictions, references):
    return {
        "exact_match": 100.0 * sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)
    }

# 4. Inference loop
preds, refs = [], []

for example in tqdm(dataset):
    question = example["question"]
    reference = example["answerText"] if isinstance(example["answerText"], str) else example["answerText"][0]

    # Chat-style prompt formatting
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers factual questions concisely."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract assistant response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if messages[-1]["content"] in full_output:
        generated_answer = full_output.split(messages[-1]["content"])[-1].strip()
    else:
        generated_answer = full_output.strip()

    generated_answer = generated_answer.replace("Assistant:", "").strip()

    preds.append(generated_answer)
    refs.append(reference.strip())

# 5. Compute and print Exact Match score
results = compute_exact_match(preds, refs)
print(f"Exact Match Score: {results['exact_match']:.2f}%")