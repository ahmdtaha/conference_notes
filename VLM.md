# VLM Review

## Architecture/Training Objectives

### 1. Contrastive Language Image Pre-training (CLIP) by OpenAI
Trains on image-text pairs crawled from internet. [arXiv](https://arxiv.org/abs/2103.00020)
![](./imgs/clip_openai.png)

### 2. A Large-scale ImaGe and Noisy-text embedding (ALIGN) by Google Research
Trains on _Noisy_ image-text pairs crawled from internet. [arXiv](https://arxiv.org/abs/2102.05918)

The _Noisy_ training datasets is the key difference between ALIGN and CLIP.

![](./imgs/align_google.png)


### 3. Bootstrapping Language-Image Pre-training (BLIP) by Salesforce
Train VLM with three different objectives to support both understanding-based tasks
or generation-based tasks. [arXiv](https://arxiv.org/abs/2201.12086)

BLIP leverages three training objectives:
1. Image-Text Contrastive (ITC) like CLIP/ALIGN
2. Image-Text Matching (ITM) which is a binary cross-entropy loss to decide whether the text-image pair matches or not. The loss is applied on the [encode] token.
3. Language Modeling (LM) which is an auto-regressive objective -- like typical V/LLM -- that generates the image caption. The caption generation starts after the [decode] token.

![](./imgs/blip_salesforce.png)

