# 🌟 Code for "Addressing Hallucinations in Language Models: Knowledge Graph Embeddings as an Additional Modality"

This repository contains the evaluation code and setup instructions to reproduce the experiments described in our paper.

---

## 💾 1. Setup and Checkpoint Download

To begin, you must download the pre-trained weights for our primary model, `Mistral-7B-KG`.

1.  Navigate to the code directory:
    ```bash
    cd kg_reduces_halu
    ```

2.  Download the archive containing the model checkpoints (including both `Mistral-7B` and `Mistral-7B-KG`):
    ```bash
    wget '[https://www.dropbox.com/scl/fi/yk35riqrumf90n3po5azz/chkpts1.tar.gz?rlkey=duylp86l0kju23xggr7zb2cu5&st=3826jrv9&dl=1](https://www.dropbox.com/scl/fi/yk35riqrumf90n3po5azz/chkpts1.tar.gz?rlkey=duylp86l0kju23xggr7zb2cu5&st=3826jrv9&dl=1)' -O chkpts.tar.gz
    ```

3.  Unpack the archive:
    ```bash
    tar -xvzf chkpts.tar.gz
    ```

## 🧪 2. HaluEval Benchmark Reproduction

The evaluation on the HaluEval benchmark is based on the original repository ([RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)) with minor code modifications for integration.

### **Evaluation Steps:**

1.  Navigate to the evaluation directory:
    ```bash
    cd HaluEval
    ```

2.  Run the evaluation script, specifying the desired task and model:
    ```bash
    python evaluate.py --task <task> --model <model>
    ```

| `--task` Argument | Available Tasks |
| :--- | :--- |
| `qa` | Question Answering |
| `summarization` | Summarization |
| `dialogue` | Dialogue |

| `--model` Argument | Available Checkpoints |
| :--- | :--- |
| `mistral7b` | The base Mistral-7B model |
| `mistral7b-kg` | Our Mistral-7B model augmented with Knowledge Graph Embeddings |

## ✅ 3. True-False Dataset Evaluation

To reproduce the model performance evaluation on the **"True-False"** dataset:

1.  Navigate to the True-False directory:
    ```bash
    cd True-False
    ```

2.  Execute the script:
    ```bash
    python evaluate.py --model <model> --kg 
    ```
    * **Note:** The optional `--kg` flag is used to select the Knowledge Graph-augmented model (`mistral7b-kg`) for evaluation.

## 📚 4. Dataset Example

A subset of the dataset described in the paper is provided in the following folder for reference:

* `dataset_example/`
