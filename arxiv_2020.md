* Exploring Simple Siamese Representation Learning **#arxiv2020**
* Dissecting Image Crops -- `PI_Reading_Grp` **#arxiv2020**
* Meta Pseudo Labels **#arxiv2020**
* Training data-efficient image transformers & distillation through attention -- `PI_Reading_Grp` :hash:industry :hash:FB **#arxiv2020**
	> Technicalities
* A Simple Semi-Supervised Learning Framework for Object Detection **#arxiv2020** :hash:industry :hash:Google Brain
* Localization Uncertainty Estimation for Anchor-Free Object Detection -- `Gaussian FCOS` **#arxiv2020**
* Are we done with ImageNet? **#arxiv2020**
* Talking-Heads Attention **#arxiv2020**
* Learned Initializations for Optimizing Coordinate-Based Neural Representations **#arxiv2020**
* Splitting Convolutional Neural Network Structures for Efficient Inference **#arxiv2020**
* Your Classifier Is Secretly An Energy Based Model And You Should Treat It Like One **#arxiv2020**
* Architecture Disentanglement for Deep Neural Networks **#arxiv2020**
* Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases **#arxiv2020**
* Supervised Contrastive Learning **#arxiv2020**
* Supermasks in superposition **#arxiv2020**
* Deep Adaptive Semantic Logic (DASL): Compiling Declarative Knowledge into Deep Neural Networks **#arxiv2020**
	> Interesting topic. Should read more about it.
* Dynamic Sampling for Deep Metric Learning **#arxiv2020**
* Watching the World Go By: Representation Learning from Unlabeled Videos **#arxiv2020**
* State-of-Art-Reviewing: A Radical Proposal to Improve Scientific Publication **#arxiv2020**
* A Simple Framework for Contrastive Learning of Visual Representations **#arxiv2020**
	> Looks simple but use a batch_size=8192, trained for 100 epochs on imagenet.
* Scaling Laws for Neural Language Models `OpenAI`
	> The relationship between loss and compute/#parameters/dataset follows power laws, i.e., L = x^\pow. These laws can identify compute optimal regime, how to scale parameters and dataset. Given more computational power, it is better to train a larger network on a small dataset (#iterations) than a smaller network on a bigger dataset (#iterations). The number of training -- non-Embedding -- parameters is more important compared to the architecture itself (width and high). It is recommended to use a small batch-size at early training stage when loss is high and a big batch-size at late stage when loss is small.
* Adaptive Self-Training For Few-Shot Neural Sequence Labeling **#arxiv2020**
* Stabilizing the Lottery Ticket Hypothesis **#arxiv2019**
* Sparse Transfer Learning via Winning Lottery Tickets **#arxiv2019**
* INTRIGUING PROPERTIES OF LEARNED REPRESENTATIONS **openreview2019-ICLR**
* Unsupervised Representation Learning by Predicting Image Rotations
* Self-EMD: Self-Supervised Object Detection without ImageNet
* MammoGANesis: Controlled Generation of High-Resolution Mammograms for Radiology Education `arXiv` `mammogram` `Medical`
* Contrastive Learning of Medical Visual Representations from Paired Images and Text `Medical`
* TResNet: High Performance GPU-Dedicated Architecture `Technicalities`
* RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification 
	> 14,368 high-resolution shelf images, on average, 37.1 objects per image, resulting in 533,633 images of individual objects. Each individual object image represents a product from in total of 2000 SKUs
* Toward transformer-based object detection `Pinterest`
	> Leverage Residual-blocks on top of ViT-features to enable Object detection with ViT-backbone.
* Scaling Laws for Neural Language Models `OpenAI`
	> The relation between loss and training resources (# parameters, training samples, GPUs) is covered by a power-law. It is better to train a large model on a small number of samples (with early stopping) than to train a small model on a large number of samples. The # parameters is more important for performance compared to the architecture itself (depth/width). Optimal Batch-size is related to loss function, with small batches at high loss and bigger batches at lower loss. Given more compute, use it to train bigger models and less on more data.