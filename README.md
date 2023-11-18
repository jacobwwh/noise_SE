# An Empirical Study on Noisy Label Learning for Program Understanding

## Requirements

pytorch

transformers

DGL

## Datasets:

Code classification: CodeNet   [Link](https://developer.ibm.com/data/project-codenet/)

Vulnerability detection: Devign   [link](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing), please follow the instructions in [link](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) for preprocessing the data.

Code summarization: TLC   [Link](https://drive.google.com/file/d/1m4uZi0hoInYxkgrSlF23EjVasSgaXOXy/view)

The samples with human evaluation results are stored in data/


## Models

Trained-from-scratch models: LSTM, GNN, Transformer (summarization)

Pre-trained models: CodeBERT, GraphCodeBERT, UniXCoder, PLBART (summarization)


## How to Run

Program classification and vulnerability detection: python run_classification_xxx.py

Code summarization: see code_sum/



