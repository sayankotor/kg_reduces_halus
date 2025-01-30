import os, json
import pandas as pd

import pickle

import warnings
warnings.filterwarnings("ignore")

path = "/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/big_graph/"

with open(path + 'name2num.pickle', 'rb') as handle:
    name2num = pickle.load(handle)
    
print("len", len(name2num), flush = True)
    
with open(path + 'embedds.pickle', 'rb') as handle:
    embedds = pickle.load(handle)
    
print("len embedds", len(embedds),flush = True)


succsess = 0
fail = 0

def get_embedding(ent):
    num = name2num.get(ent, -111)
    emb = embedds.get(num, -111)
    return emb

def get_qtext_and_embeddings(json_text):
    special_symbol = "#Q"
    
    generated_text = json_text['text']
    ents = []
    ents_numbers = []
    ents_embeddings = []
    set_of_seen = set()
    for ind, elem in enumerate(json_text['ents']['ents']):
        if (elem is not None and elem not in set_of_seen):
            ents_numbers.append(elem)
            ents.append(json_text['ents']['ent_names'][ind])
            #print (elem, json_text['ents']['ent_names'][ind])
            set_of_seen.add(elem)
            #ents_embeddings.apppend(get_embedding(elem))

    
    ents_existed = []
    ents_num_existed = []
    list_of_embs = []
    
    for ind, ent in enumerate(ents):
        #pos = generated_text.find(ent)
        #print (pos)
        res = [i for i in range(len(generated_text)) if generated_text.startswith(ent, i)]
        if (len(res) > 0):
            emb = get_embedding(ents_numbers[ind])
            if (emb != -111):
                global succsess
                succsess = succsess + 1
                list_of_embs.append(emb)
                ents_existed.append(ent)
                ents_num_existed.append(ents_numbers[ind])
                for pos in res[::-1]:
                    generated_text = generated_text[:pos] + special_symbol + generated_text[pos:]
            else:
                #print (ent, ents_numbers[ind])
                global fail
                fail = fail + 1
     
    #print (fail, succsess)
    #list_of_embs = [get_embedding(ent) for ent in ents_num_existed]
    return generated_text, ents_existed, ents_num_existed, list_of_embs


# this finds our json files
path_to_json = '/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/big_wikipedia_omonims_new'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

print ("number of files", len(json_files), flush = True)

# here I define my pandas Dataframe with the columns I want to get from the json
pd_data = pd.DataFrame(columns=['text', 'text_with_Q', 'entities', 'entity_numbs', 'entity_embs'])
#data = dict.fromkeys(['text', 'text_with_Q', 'entities', 'entity_numbs', 'entity_embs'])
for ind, json_file in enumerate(json_files):

    with open(os.path.join(path_to_json + "/" + json_file)) as js:
        json_text = json.load(js)
        text = json_text["text"]
        generated_text, list_of_ents, list_of_num_ents, list_of_embs = get_qtext_and_embeddings(json_text)
        
        pd_data = pd_data.append({'text':text, 'text_with_Q': generated_text, 'entities':list_of_ents, 'entity_numbs':list_of_num_ents, 'entity_embs':list_of_embs}, ignore_index=True)
        
        if (ind%50000 == 0):
            print ("ind", ind, flush = True)
            print ("fail, succsess", fail, succsess, flush = True)
            
        if (ind%100000 == 0 or ind == 3276930):
            # save piece
            print ("len(pd_data)", len(pd_data),flush = True)
            pd_data.to_csv(str(ind)+'.csv', index=False) 
            pd_data = pd.DataFrame(columns=['text', 'text_with_Q', 'entities', 'entity_numbs', 'entity_embs'])
            #data = dict.fromkeys(['text', 'text_with_Q', 'entities', 'entity_numbs', 'entity_embs'])
            

