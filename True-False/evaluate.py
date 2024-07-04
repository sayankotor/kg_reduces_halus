#from gen_answers_llama import get_answ
#from gen_answers_llama3 import get_answ_llama3
from gen_answers_mistral import get_answ_mistral
import random
import openai
import time
import json
import argparse

import datasets


dataset = datasets.load_dataset("pminervini/true-false")



def evaluate(subdataset, model, is_kg = False):
    data = dataset[subdataset]
    correct = 0
    for ind, elem in enumerate(data):
        statement = data[ind]['statement']
        label = data[ind]['label']
        if (model == 'llama'):
            answ = get_answ(instruction, statement, is_kg)
        elif (model == 'llama3'):
            #instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + instruction + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            #statement = data[ind]['statement']# + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            answ = get_answ_llama3(instruction, statement, is_kg)
        elif (model == 'mistral'):
            answ = get_answ_mistral(instruction, statement, is_kg)
        else:
            raise NameError('Model name is not supported.')
        print ("\n\n" + statement+ "\n" + str(label) + "\n" +answ + "\n\n")
        if ("True" in answ and "False" in answ):
            print('sample {} fails......'.format(ind))
            continue
        elif ("true" in answ and "false" in answ):
            print('sample {} fails......'.format(ind))
            continue
            
        elif label == 1:
            if ("True" in answ or " correct" in answ or "true" in answ or " accurate" in answ):
                correct+=1
                print('sample {} succeed......'.format(ind))
                continue
            else:
                print('sample {} fails......'.format(ind))
                continue
        elif label == 0:
            if ("False" in answ or "incorrect" in answ or "false" or "unaccurate" in answ):
                correct+=1
                print('sample {} succeed......'.format(ind))
                continue
            else:
                print('sample {} fails......'.format(ind))
                continue
        else:
            print('sample {} fails......'.format(ind))
            continue
    
    return correct/len(data)

import json
import os
output_path = os.path.dirname(os.path.abspath(__file__))

with open('instruction.txt', 'r') as file:
    instruction = file.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="True False check")

    parser.add_argument("--kg", action='store_true', help="should we add knowledge graphs to generate answer")
    parser.add_argument("--model", default="llama", help="model name")
    args = parser.parse_args()
    
    results = {}
    
    with open(output_path + '/result_llama2_kg_comp.json', 'w') as fp:
            json.dump(results, fp)
    
    subdatasets = ["companies"]#, "facts", "inventions", "elements", "cities", "animals", "generated", "cieacf"] #"facts"
    for subdataset in subdatasets:
        acc = evaluate(subdataset, args.model, args.kg)
        results[subdataset] = acc
        
        with open(output_path + '/result_llama2_kg_comp.json', 'w') as fp:
            json.dump(results, fp)
        
    
    
