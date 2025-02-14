import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
from huggingface_hub import hf_hub_download


DEVICE = "cuda:0"



from huggingface_hub import login
login(token='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')

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
from huggingface_hub import hf_hub_download


DEVICE = "cuda:0"
PROMPT = "This is a dialog with AI assistant.\n"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map=DEVICE)

import os


path = os.getcwd()
path_parent = os.path.abspath(os.path.join(path, os.pardir))
path_root = os.path.dirname(path_parent)

projection = torch.load(path_root + "/chkpts/projection_qa", map_location=DEVICE)
start_emb = torch.load(path_root+ "/chkpts/SOI2_qa.pt", map_location=DEVICE)
end_emb = torch.load(path_root+ "/chkpts/EOI2_qa.pt", map_location=DEVICE)

# Load embedding encoder

from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = path_root+"/chkpts/graphRoberta_v1"
projector_path = path_root+"/chkpts/projector_v1"


tokenizer_emb = AutoTokenizer.from_pretrained(model_path)
model_emb = AutoModelForMaskedLM.from_pretrained(model_path).to(DEVICE)
projector = torch.load(projector_path).to(DEVICE)

def text2graph_emb(text):
    with torch.no_grad():
        input_ids = tokenizer_emb.encode(text, return_tensors="pt").to(DEVICE)
        output = model_emb(input_ids, output_hidden_states=True)
        pooled_emb = output["hidden_states"][-1].mean(1)
        projected_emb = projector(pooled_emb)
    
    return projected_emb

bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids

gen_params = {
        "do_sample": False,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
    }

def ask_mistral_with_prompt(question, prompt):
    print ("ask_mistral_with_prompt", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    question_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)
    embeddings = prompt_embeddings
    embeddings = torch.cat(
            [
                prompt_embeddings,
                question_embeddings,
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts


def ask_mistral_with_kg_embeddings(question, prompt):
    print ("ask_mistral_with_kg embs", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    question_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)
    ent_embs = text2graph_emb(question).to(torch.bfloat16)
    try:
        #ent_embs = torch.tensor(ent_embs).to(device=DEVICE, dtype=model.dtype)
        m = ent_embs.mean(1, keepdim=True)
        s = ent_embs.std(1, unbiased=False, keepdim=True)
        ent_embs -= m
        ent_embs /= s
    except:
        print (ent_embs.shape)
    projected_kg_embeddings = projection(ent_embs[:,None,:])
    #print (question_embeddings.shape, start_emb[None, None, ...].shape, projected_kg_embeddings.shape)
    embeddings = torch.cat(
               [
                    prompt_embeddings,
                    start_emb[None, None, ...],
                    projected_kg_embeddings,
                    end_emb[None, None, ...],
                    question_embeddings
                ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts


def ask_mistral_with_kg(question, prompt, ent_embs):
    print ("ask_mistral_with_kg", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    question_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)
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
        print (question_embeddings.shape, start_emb[None, None, ...].shape, projected_kg_embeddings.shape, flush = True)
        embeddings = torch.cat(
                   [
                        prompt_embeddings,
                        start_emb[None, None, ...],
                        projected_kg_embeddings,
                        end_emb[None, None, ...],
                        question_embeddings
                    ],
                dim=1,
            ).to(dtype=torch.bfloat16, device=DEVICE)
    else:
        embeddings = torch.cat(
                   [
                        prompt_embeddings,
                        question_embeddings
                    ],
                dim=1,
            ).to(dtype=torch.bfloat16, device=DEVICE)
        
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

## Summarization

def mistral_summary(prompt, summary):
    print ("mistral_summary", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    summary_ids = tokenizer.encode(summary, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    summary_embeddings = model.model.embed_tokens(summary_ids).to(torch.bfloat16)
    #embeddings = prompt_embeddings
    embeddings = torch.cat(
            [
                prompt_embeddings,
                summary_embeddings,
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 40)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

def split_doc_to_chunks(doc, model, max_seq_len = 512):
    if (len(doc)) >= max_seq_len:
        n_chunks = len(doc)//max_seq_len
        embs = []
        for i in range(n_chunks):
            emb = text2graph_emb(doc[i:i+max_seq_len]).to(torch.bfloat16)
            embs.append(emb)
        embs = torch.cat(embs)
        print (embs.shape)
    else:
        embs = text2graph_emb(doc)
    return embs

def mistral_summary_kg_embeddings(prompt, document, summary):
    print ("mistral_summary_with_kg", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    print ("1")
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    print ("2")
    summary_ids = tokenizer.encode(summary, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    print ("3")
    summary_embeddings = model.model.embed_tokens(summary_ids).to(torch.bfloat16)
    print ("4")
    ent_embs = split_doc_to_chunks(document, text2graph_emb)
    print (summary_embeddings.shape, start_emb[None, None, ...].shape, ent_embs.shape, flush = True)
    try:
        ent_embs = torch.tensor(ent_embs).to(device=DEVICE, dtype=model.dtype)
        m = ent_embs.mean(1, keepdim=True)
        s = ent_embs.std(1, unbiased=False, keepdim=True)
        ent_embs -= m
        ent_embs /= s
    except:
        print ("exception", len(ent_embs), ent_embs)
    projected_kg_embeddings = projection(ent_embs[None,:,:])
    print (summary_embeddings.shape, start_emb[None, None, ...].shape, projected_kg_embeddings.shape, flush = True)
    embeddings = torch.cat(
                [
                    prompt_embeddings,
                    start_emb[None, None, ...],
                    projected_kg_embeddings,
                    end_emb[None, None, ...],
                    summary_embeddings
                ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
        
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts
    
    
def mistral_dialog(prompt, response):
    print ("mistral_dialog", flush = True)
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    response_embeddings = model.model.embed_tokens(response_ids).to(torch.bfloat16)
    #embeddings = prompt_embeddings
    embeddings = torch.cat(
            [
                prompt_embeddings,
                response_embeddings,
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

def mistral_dialog_kg_embeddings(instruction, dialog, response):
    print ("mistral_dialog_kg", flush = True)
    prompt = instruction + "\n\n#Dialogue History#: " + dialog
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    response_embeddings = model.model.embed_tokens(response_ids).to(torch.bfloat16)
    #embeddings = prompt_embeddings
    ent_embs = split_doc_to_chunks(dialog, text2graph_emb)
    try:
        ent_embs = torch.tensor(ent_embs).to(device=DEVICE, dtype=model.dtype)
        m = ent_embs.mean(1, keepdim=True)
        s = ent_embs.std(1, unbiased=False, keepdim=True)
        ent_embs -= m
        ent_embs /= s
    except:
        print ("exception", len(ent_embs), ent_embs)
    #projected_kg_embeddings = projection(ent_embs[None,:,:])
    projected_kg_embeddings = projection(ent_embs[:,None,:])
    print (prompt_embeddings.shape, start_emb[None, None, ...].shape, projected_kg_embeddings.shape, flush = True)
    embeddings = torch.cat(
               [
                    prompt_embeddings,
                    start_emb[None, None, ...],
                    projected_kg_embeddings,
                    end_emb[None, None, ...],
                    response_embeddings
                ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

def dialogue_mistral_with_knowledge(prompt, response, knowledge):
    prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
    response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
    response_embeddings = model.model.embed_tokens(response_ids).to(torch.bfloat16)
    #embeddings = prompt_embeddings
    ent_embs = text2graph_emb(knowledge).to(torch.bfloat16)
    try:
        ent_embs = torch.tensor(ent_embs).to(device=DEVICE, dtype=model.dtype)
        m = ent_embs.mean(1, keepdim=True)
        s = ent_embs.std(1, unbiased=False, keepdim=True)
        ent_embs -= m
        ent_embs /= s
    except:
        print ("exception", len(ent_embs), ent_embs)
    #projected_kg_embeddings = projection(ent_embs[None,:,:])
    projected_kg_embeddings = projection(ent_embs[:,None,:])
    print (prompt_embeddings.shape, start_emb[None, None, ...].shape, projected_kg_embeddings.shape, flush = True)
    embeddings = torch.cat(
               [
                    prompt_embeddings,
                    start_emb[None, None, ...],
                    projected_kg_embeddings,
                    end_emb[None, None, ...],
                    response_embeddings
                ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 30)
    #out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts
    