* Learning Transferable Visual Models From Natural Language Supervision **#openai2021**
* LambdaNetworks: Modeling long-range Interactions without Attention **#openreview2021** -- Seems like a good idea, poorly written.
* Re-labeling ImageNet:from Single to Multi-Labels, from Global to Localized Labels **#arXiv2021** -- `PI_Reading_Grp`
* Bottleneck Transformers for Visual Recognition **#arXiv2021** -- `PI_Reading_Grp` Technicalities
* Towards General Purpose Vision Systems **#arXiv2021** -- `PI_arXiv_Grp` Technicalities
* Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision **#arXiv2021** -- `PI_arXiv_Grp` Technicalities
* Learning Rate Grafting: Transferability of Optimizer Tuning
	> Optimize using two optimizers: one optimizer provides the magnitude while the other provides the direction.
* MLP-Mixer: An all-MLP Architecture for Vision `Arch_Design`
* FNet: Mixing Tokens with Fourier Transforms  `Arch_Design`
* Learning Open-World Object Proposals without Learning to Classify `Detection`
* nnFormer: Interleaved Transformer for Volumetric Segmentation `Detection`
* Early Convolutions Help Transformers See Better  `Transformer`
* Escaping the Big Data Paradigm with Compact Transformers `Transformer`
* Efficient Training of Visual Transformers with Small-Size Datasets  `Transformer`
* VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning `SSL`
* Swin transformer: Hierarchical vision transformer using shifted windows `Transformer`
* ResNet strikes back: An improved training procedure in timm 
* Do Self-Supervised and Supervised Methods Learn Similar Visual Representations?
* Self-supervised pretraining of visual features in the wild `FAIR`
* Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers `Google/DeepMind`
* ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases `FAIR` `Transformer`
* Open-Set Recognition: A Good Closed-Set Classifier is All You Need
* Extrapolating from a Single Image to a Thousand Classes using Distillation
* Are Large-scale Datasets Necessary for Self-Supervised Pre-training? `META` `arXiv`
* Zero-Shot Text-to-Image Generation `arXiv` `dall-e` `OpenAI` `dVAE` `Transformers`
* MPViT : Multi-Path Vision Transformer for Dense Prediction `arXiv` `Transformer`
* Masked Feature Prediction for Self-Supervised Visual Pre-Training `Video` `SSL` `Meta`
* DeepViT: Towards Deeper Vision Transformer `ViT`
	> Nice analysis
* SCD: A Stacked Carton Dataset for Detection and Segmentation
	> 16,136 (8401 online + 7735 collected) with 250,000 instance masks (semantic segmentation).
* Deep learning through the lens of example difficulty `Google` `Nice`
	> the prediction depth
* An Attention Free Transformer `Apple`
	> I am not sure if this scales beyond tiny models/inputs.
* SA-GD: Improved Gradient Descent Learning Strategy with Simulated Annealing 
	> Apply gradient ascent stochastically at later stage of training to escape local minima/saddle points.
* Benchmarking detection transfer learning with vision transformers `Meta`
	> MAE and BEiT pre-training benefit detection.
* Rethinking Self-Supervised Learning: Small is Beautiful
	> The paper promotes small dataset/resolution/arch learning to achieve competitive performance with limited compute. Not sure if the paper conclusions/findings are valid, but the SSL pre-training on combined small and large resolutions makes sense.
* Vision Transformer Pruning
	> Pruning ViT by learning an importance score for every dimension d durnig training. unimportant dimensions are pruned and a final fine-tuned stage is performance to recover lost accuracy. An l1-regularization is applied on importance score.
* Bad Global Minima Exist and SGD Can Reach Them `Mila` `Canada` `Greece` `USA` `Academia`
	> Bad minima can be reached by pre-training on noisy dataset (with noisy label) then using vanilla SGD without any augmentation or reguralization or momentum. During noisy pre-training, the network learns a complex features and a complex decision boundary. During the second training stage, SGD stays close to inferior initialization -- it doesn't generalize to test/val splits.