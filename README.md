# Weigh Your Own Words: Improving Hate Speech Counter Narrative Generation via Attention Regularization

This repository contains the code to replicate our experiments.

## Getting Started

Create a new Python environment and install the required packages with

```bash
pip install -r requirements
```

## Fine-Tuning 

Use [finetune_gpt2.py](./finetune_gpt2.py) to fine-tune a GPT2 model using KLAR or EAR.

## Attention Regularization

Please refer to the `src` folder to find our custom GPT2 implementation and code to compute KLAR and EAR attention-based losses. 

## Reference

TBD
