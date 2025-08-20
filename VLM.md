# VLM Review

## Architecture/Training Objectives

### 1. Contrastive Language Image Pre-training (CLIP) by OpenAI [arXiv](https://arxiv.org/abs/2103.00020)
Trains on image-text pairs crawled from internet. 
![](./imgs/clip_openai.png)

### 2. A Large-scale ImaGe and Noisy-text embedding (ALIGN) by Google Research [arXiv](https://arxiv.org/abs/2102.05918)
Trains on _Noisy_ image-text pairs crawled from internet. 

The _Noisy_ training datasets is the key difference between ALIGN and CLIP.

![](./imgs/align_google.png)


### 3. Bootstrapping Language-Image Pre-training (BLIP) by Salesforce [arXiv](https://arxiv.org/abs/2201.12086)
Train VLM with three different objectives to support both understanding-based tasks
or generation-based tasks. 

BLIP leverages three training objectives:
1. Image-Text Contrastive (ITC) like CLIP/ALIGN
2. Image-Text Matching (ITM) which is a binary cross-entropy loss to decide whether the text-image pair matches or not. The loss is applied on the [encode] token.
3. Language Modeling (LM) which is an auto-regressive objective -- like typical V/LLM -- that generates the image caption. The caption generation starts after the [decode] token.

![](./imgs/blip_salesforce.png)

### 4. Bootstrapping Language-Image Pre-training-v2 (BLIP-2) by Salesforce [arXiv](https://arxiv.org/abs/2301.12597)
Train VLM with minimal computatinal cost (both frozen vision and LLM). 

The paper trains a Q-former between a frozne vision encoder and an LLMs. The Q-former objective is to bridge the gap between the vision and text representation. To train Q-former, the paper propose a two-stage pipeline: The first stage uses BLIP three-objectives: ITC, ITM, and LM. The first stage objective is to align vision and text represnetations. The second stage is solely a generative language modeling (LM) stage. The second stage objective is to bootstrap vision-to-language generative learning. 

Key observations: 
1. The Q-former can be regarded as a non-linear projection layer between the vision and text embedding which is a common approach in recent literature when training VLMs. 
2. BLIP-2 propose a two-stage training pipeline which was a new thing in 2023. Nowadays (2025), all VLMs employ a two-stage training pipeline but of course within different training objectives. The current two-stage training pipeline is usually pre-training on a large corpus of unlabeled data, before "fine-tuning" on instruction-following data.
 

![](./imgs/blip2_1_salesforce.png)
![](./imgs/blip2_2_salesforce.png)


## Pre-training Datasets

### 1.  Conceptual Captions 12M (CC12M) by Google [arXiv](https://arxiv.org/abs/2102.08981)

A dataset with 12 million image-text pairs specifically meant to be used for visionand-language pre-training.

![](./imgs/cc12m_google.png)


## Benchmarks

### 1. Science Question Answering (ScienceQA) by AI2 [arXiv](https://arxiv.org/abs/2209.09513)

Multimodal multiple choice questions with diverse science topics and annotations of their answers with corresponding lectures and explanations. 


![](./imgs/science_ai2.png)