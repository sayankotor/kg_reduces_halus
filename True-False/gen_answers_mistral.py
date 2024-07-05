DEVICE = "cuda:0"

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
from huggingface_hub import hf_hub_download

DEVICE = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map=DEVICE)


projection = torch.load("/ckpts/projection2", map_location=DEVICE)
start_emb = torch.load("/ckpts/SOI2.pt", map_location=DEVICE)
end_emb = torch.load("/ckpts/EOI2.pt", map_location=DEVICE)

# Load embedding encoder

from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = "/ckpts/graphRoberta_v1"
projector_path = "/ckpts/projector_v1"


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

def get_answ_mistral(instruction, statement, is_kg):
    #print ("is kg", is_kg)
    if (not is_kg):
        #print ("ask_mistral_with_prompt", flush = True)
        prompt = instruction + "\n#Statement#: " + statement + "\n#Your Judgement#:"
        prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        embeddings = prompt_embeddings
        out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 20)
        #out = out[:, 1:]
        generated_texts = tokenizer.batch_decode(out)[0]
        return generated_texts
    else:
        #print ("add mistral with kg embeddings")
        prompt_ids = tokenizer.encode(f"{instruction}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        ent_embs = text2graph_emb(statement).to(torch.bfloat16)
        statement = "\n#Statement#: " + statement + "\n#Your Judgement#:"
        statement_ids = tokenizer.encode(statement, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        statement_embeddings = model.model.embed_tokens(statement_ids).to(torch.bfloat16)
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
                        statement_embeddings
                    ],
                dim=1,
            ).to(dtype=torch.bfloat16, device=DEVICE)
        out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 20)
        #out = out[:, 1:]
        generated_texts = tokenizer.batch_decode(out)[0]
        return generated_texts
        

        
         