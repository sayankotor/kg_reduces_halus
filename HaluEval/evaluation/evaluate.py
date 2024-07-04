import random
import openai
import time
import json
import argparse
#import tiktoken

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load entities embeddings for QA

import pickle
with open('/home/jovyan/shares/SR004.nfs2/chekalina/check_halu/column_with_embs.pkl', 'rb') as f:
    column_with_embs = pickle.load(f)

with open('/home/jovyan/shares/SR004.nfs2/chekalina/check_halu/column_with_entss.pkl', 'rb') as f:
    column_with_entites = pickle.load(f)



openai.api_key = 'sk-'

def get_qa_response(model, question, answer, instruction):
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "} 
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response

def get_qa_response_private(model, question, answer, instruction):
    if ("llama3" in model):
        prompt = instruction 
        #question_full = "\n\n#Question#: " + question + "\n#Answer#: " + answer +"\n#Your Judgement#:"
        #prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + prompt +"<|eot_id|><|start_header_id|>user<|end_header_id|>"
        #question_full = "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    else:
        prompt = instruction 
        question_full = "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    #print ("\n\n\n\n")
    #print ("prompt", prompt)
    while True:
        try:
            if model == "mistral7b":
                res = ask_mistral_with_prompt(question, prompt + question_full)
            elif model == "mistral7b-kg":
                res = ask_mistral_with_kg_embeddings(question_full, prompt)
            elif model == "llama27b":
                res = ask_llama_with_prompt(question, prompt + question_full)
            elif model == "llama27b-kg":
                res = ask_llama_with_kg_embeddings(question_full, prompt)
            elif model == "llama3":
                res = ask_llama3_with_prompt(question_full, prompt)
            elif model == "llama3-kg":
                res = ask_llama3_with_kg_embeddings(question_full, prompt)
            break
        except Exception as e: 
            print(e)
            time.sleep(60)

    return res

def get_qa_response_kg(model, question, answer, instruction, embs):
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "} 
    ]
    if ("llama3" in model):
        prompt = "<|begin_of_text|>" + instruction
    else:
        prompt = instruction 
    print ("get_qa_response_kg", flush = True)
    question_full = "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    #print ("\n\n\n\n")
    #print ("prompt", prompt)
    print ("\n")
    while True:
        try:
            if ('mistral' in model):
                res = ask_mistral_with_kg(question_full, prompt, embs)
            else:
                res = ask_llama_with_kg(question_full, prompt, embs)
            break
        except Exception as e: 
            print(e)
            time.sleep(60)

    return res

    
def get_dialogue_response_private(model, dialog, response, instruction):
    if ("llama3" in model):
        prompt = "<|begin_of_text|>" + instruction + "\n\n#Dialogue History#: " + dialog
    else:
        prompt = instruction + "\n\n#Dialogue History#: " + dialog
    response = "\n#Response#: " + response + "\n#Your Judgement#:"
    print (model, "\n")
    while True:
        try:
            if model == "mistral7b":
                res = mistral_dialog(prompt, response)
            elif model == "mistral7b-kg":
                res = mistral_dialog_kg_embeddings(instruction, dialog, response)
            elif model == "llama27b":
                res = llama_dialog(prompt, response)
            elif model == "llama27b-kg":
                res = llama_dialog_kg_embeddings(instruction, dialog, response)
            elif model == "llama3":
                res = llama3_dialog(prompt, response)
            elif model == "llama3-kg":
                res = llama3_dialog_kg_embeddings(instruction, dialog, response)
            break
        except Exception as e: 
            print(e)
            time.sleep(60)

    return res

def get_dialogue_response_knowledge(model, dialog, response, instruction, knowledge):
    prompt = instruction
    response = "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    if 'mistral' in model:
        res = dialogue_mistral_with_knowledge(prompt, response, knowledge)
    else:
        res = dialogue_llama_with_knowledge(prompt, response, knowledge)
    return res
        


def get_dialogue_response(model, dialog, response, instruction):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def num_tokens_from_message(message, model="mistral7b"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt1, prompt2


def get_summarization_response(model, document, summary, instruction):
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if model == "davinci":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except Exception as e: 
            print(e)
            time.sleep(60)

    return res

def get_summarization_response_private(model, document, summary, instruction):
    prompt = "<|begin_of_text|>" + instruction + document #+ "<|start_header_id|>user<|end_header_id|>\n\n#Document#: " + document
    summary = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    #<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    #prompt = instruction + "/n#Document#: " + document
    #summary = "\n#Summary#: " + summary + "\n#Your Judgement#:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    #print ("\n\n\n\n")
    print ("prompt", prompt)
    #print ("\n")
    

    #if 'mistral' in model:
        #prompt,summary = truncate_message(prompt, summary)
    while True:
        try:
            if model == "mistral7b":
                res = mistral_summary(prompt, summary)
            elif model == "mistral7b-kg":
                res = mistral_summary_kg_embeddings(prompt, document, summary)
            elif model == "llama27b":
                res = llama_summary(prompt, summary)
            elif model == "llama27b-kg":
                res = llama_summary_kg_embeddings(prompt, document, summary)
            elif model == "llama3":
                res = llama3_summary(prompt, document, summary)
            elif model == "llama3-kg":
                print ("llama3 kg")
                res = llama3_summary_kg_embeddings(prompt, document, summary)
            break
        except Exception as e: 
            print(e)
            time.sleep(60)

    return res



def evaluation_qa_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        #answers = load_qa_response(path)
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]
            embs = column_with_embs[i]
            ents = column_with_entites[i]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            if (model in ['mistral7b', 'mistral7b-kg', 'llama27b', 'llama27b-kg', 'llama3', 'llama3-kg']):
                print ("evaluation_prvate", flush = True)
                ans = get_qa_response_private(model, question, answer, instruction)
            elif (model in ['mistral7b-kg-noencoder', 'llama-kg-noencoder']):
                print ("evaluation_qa_dataset", flush = True)
                embs = [emb for emb in embs if emb != -111]
                ans = get_qa_response_kg(model, question, answer, instruction, embs)
            else:
                ans = get_qa_response(model, question, answer, instruction)
            print ("\n answ", ans)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
                print('sample {} success......'.format(i))
            else:
                incorrect += 1
                print('sample {} fails......'.format(i))

            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/len(data)))


def evaluation_dialogue_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"
            
            
            if (model in ['mistral7b', 'mistral7b-kg', 'llama27b', 'llama27b-kg','llama3', 'llama3-kg']):
                print ("evaluation_d1_dataset", flush = True)
                ans = get_dialogue_response_private(model, dialog, response, instruction)
            elif (model in ['mistral7b-kg-noencoder', 'llama-kg-noencoder']):
                print ("evaluation_d2_dataset", flush = True)
                embs = [emb for emb in embs if emb != -111]
                ans = get_dialogue_response_kg(model, dialog, response, instruction, embs)
            elif model in ['mistral-encode-knowlegde', 'llama-encode-knowlegde']:
                ans = get_dialogue_response_knowledge(model, dialog, response, instruction, knowledge)
            else:
                print ("else", flush = True)
                ans = get_dialogue_response(model, question, answer, instruction)
            print ("\n answ", ans)
            ans = ans.replace(".", "")
            
            
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
                print('sample {} success......'.format(i))
            else:
                incorrect += 1
                print('sample {} fails......'.format(i))

            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def evaluation_summarization_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            #ans = get_summarization_response(model, document, summary, instruction)
            #print ("document", document)
            #print ("summary", summary)
            #print ("instruction", instruction)
            #print ("\n\n\n")
            print (model)
            if (model in ['mistral7b', 'mistral7b-kg', 'llama27b', 'llama27b-kg','llama3', 'llama3-kg']):
                ans = get_summarization_response_private(model, document, summary, instruction)
            #elif (model in ['mistral7b-kg-noencoder', 'llama-kg-noencoder']): No opprotuniti to do this branch for summarization
                #print ("evaluation_qa_dataset", flush = True)
                #embs = [emb for emb in embs if emb != -111]
                #ans = get_summarization_response_kg(model, document, summary, instruction)
            else:
                ans = get_summarization_response(model, document, summary, instruction)
            print ("\n answ", ans)
            
            
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
                print('sample {} success......'.format(i))
            else:
                incorrect += 1
                print('sample {} fails......'.format(i))

            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="davinci", help="model name")
    args = parser.parse_args()
    
    print (args.model, args.task)

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()
    
    model = args.model
    
    #from ask_mistral import ask_mistral_with_prompt, ask_mistral_with_kg_embeddings, ask_mistral_with_kg, mistral_summary_kg_embeddings, mistral_summary, mistral_dialog_kg_embeddings, mistral_dialog, dialogue_mistral_with_knowledge
#from ask_llama27 import ask_llama_with_prompt, ask_llama_with_kg_embeddings, ask_llama_with_kg, llama_summary_kg_embeddings, llama_summary, llama_dialog, llama_dialog_kg_embeddings, dialogue_llama_with_knowledge
    if ("llama3" in model):
        from ask_llama3 import ask_llama3_with_prompt, ask_llama3_with_kg_embeddings, ask_llama3_with_kg, llama3_dialog, llama3_dialog_kg_embeddings, llama3_summary, llama3_summary_kg_embeddings
    elif ("mistral" in model):
        from ask_mistral import ask_mistral_with_prompt, ask_mistral_with_kg_embeddings, ask_mistral_with_kg, mistral_summary_kg_embeddings, mistral_summary, mistral_dialog_kg_embeddings, mistral_dialog, dialogue_mistral_with_knowledge
    elif ("llama" in model):
        from ask_llama27 import ask_llama_with_prompt, ask_llama_with_kg_embeddings, ask_llama_with_kg, llama_summary_kg_embeddings, llama_summary, llama_dialog, llama_dialog_kg_embeddings, dialogue_llama_with_knowledge
    
    
    output_path = "{}/{}_{}_results.json".format(args.task, args.task, args.model)

    data = "../data/{}_data.json".format(args.task)

    if args.task == "qa":
        evaluation_qa_dataset(model, data, instruction, output_path)
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(model, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(model, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
