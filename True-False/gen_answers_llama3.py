import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn

DEVICE = "cuda:0"
PROMPT = "This is a dialog with AI assistant.\n"

#model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b",torch_dtype=torch.bfloat16, device_map=DEVICE)
#tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct",torch_dtype=torch.bfloat16, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
model.eval()


projection = torch.load("/ckpts/projection_llama3_qa1", map_location=DEVICE)
start_emb = torch.load("/ckpts/SOI2_llama3_qa1.pt", map_location=DEVICE)
end_emb = torch.load("/ckpts/EOI2_llama3_qa1.pt", map_location=DEVICE)

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

def get_answ_llama3(instruction, statement, is_kg):
    print ("llama3 is kg", is_kg)
    if (not is_kg):
        print ("ask_llama_with_prompt", flush = True)
        #prompt = "\n#Statement#: " + statement + "\n#Your Judgement#:"
        prompt = instruction + "\n#Statement#: " + statement + "\n#Your Judgement#:"
        prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        embeddings = prompt_embeddings
        out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 40)
        #out = out[:, 1:]
        generated_texts = tokenizer.batch_decode(out)[0]
        return generated_texts
    else:
        print ("add kg embeddings")
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
        projected_kg_embeddings = projection(ent_embs[None,:,:])
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
        out = model.generate(inputs_embeds=embeddings, **gen_params, max_new_tokens = 40)
        #out = out[:, 1:]
        generated_texts = tokenizer.batch_decode(out)[0]
        return generated_texts
        
