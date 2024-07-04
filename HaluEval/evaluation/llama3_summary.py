DEBUG:urllib3.connectionpool:Starting new HTTPS connection (1): huggingface.co:443
Starting new HTTPS connection (1): huggingface.co:443
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/config.json HTTP/1.1" 200 0
Loading checkpoint shards:   0%|                                                                                                                                                                                                                               | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|█████████████████████████████████████████████████████▊                                                                                                                                                                 | 1/4 [00:01<00:03,  1.27s/it]Loading checkpoint shards:  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 2/4 [00:01<00:01,  1.07it/s]Loading checkpoint shards:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                     | 3/4 [00:02<00:00,  1.18it/s]Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.65it/s]Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.36it/s]
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/generation_config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/generation_config.json HTTP/1.1" 200 0
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /unsloth/llama-3-8b-Instruct/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
llama3
Traceback (most recent call last):
  File "/home/jovyan/shares/SR004.nfs2/chekalina/HaluEval/evaluation/evaluate.py", line 249, in get_summarization_response
    res = openai.Completion.create(
  File "/home/jovyan/.mlspace/envs/bench/lib/python3.9/site-packages/openai/lib/_old_api.py", line 39, in __call__
    raise APIRemovedInV1(symbol=self._symbol)
openai.lib._old_api.APIRemovedInV1: 

You tried to access openai.Completion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. 

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/jovyan/shares/SR004.nfs2/chekalina/HaluEval/evaluation/evaluate.py", line 532, in <module>
    evaluation_summarization_dataset(model, data, instruction, output_path)
  File "/home/jovyan/shares/SR004.nfs2/chekalina/HaluEval/evaluation/evaluate.py", line 465, in evaluation_summarization_dataset
    ans = get_summarization_response(model, document, summary, instruction)
  File "/home/jovyan/shares/SR004.nfs2/chekalina/HaluEval/evaluation/evaluate.py", line 256, in get_summarization_response
    except exception as e:
NameError: name 'exception' is not defined
