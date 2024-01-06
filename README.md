# Simple Chatbots based on Existing Open-sourced LLMs
## Introduction
This repository provides some simple scripts to realize basic chatbot functionalities based on [Phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) and [Tinyllama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) models.

## Requirements
* PyTorch >=2.0
* Transformers >=4.36

## How to use
```
python chat_tinyllama_gui.py # or
python chat_phi2_gui.py
```
Since the scripts are so short, all hyper-parameters can be easily located by manually checking.

## References
* [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
* [Microsoft-Phi-2](https://huggingface.co/microsoft/phi-2)