* Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering **#aaai2019**
* Weighted Channel Dropout for Regularization of Deep Convolutional Neural Network **#aaai2019**
* Augmenting neural networks with first- order logic **#acl2019**
* EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks `ACL2009`
	> propose simple augmentation approaches for text dataset. This is particularly useful for small datasets (e.g., medical report datasets).
* Convolutional networks with adaptive inference graphs **#ijcv2019** -- `Nice paper`
* Res2net: A new multi-scale backbone architecture **pami2019**
* Measuring the Effects of Data Parallelism on Neural Network Training **jmlr2019** `Nice` `Analysis` `Google`
* Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization **asplos2019**
* Micro-Batch Training with Batch-Channel Normalization and Weight Standardization **#arxiv2019** `WS` `BCN`
* RoBERTa: A Robustly Optimized BERT Pretraining Approach `Optimized` `BERT` `Meta` **#arxiv2019**
* Generating Diverse High-Fidelity Images with VQ-VAE-2 **#arxiv2019**
* Fast Transformer Decoding: One Write-Head is All You Need **#arxiv2019** `Google`
	> Tackle transformer inference cost problem. Intsead of multiple heads for query, key, value, the paper propose multiple-heads for the query only, i.e., a single head for key and value.
* On Mutual Information Maximization for Representation Learning **#arxiv2019**
* Faster Neural Network Training with Data Echoing **#arxiv2019** -- `Nice paper`
* Measuring Dataset Granularity **#arxiv2019**
* Objects as Points **#arxiv2019**
* On Empirical Comparisons of Optimizers for Deep Learning `Nice` `Google Brain` **#arxiv2019**
* Gacnn: Training Deep Convolutional Neural Networks With Genetic Algorithm **#NeuralEvolutionaryComputing2019**
* Energy and policy considerations for deep learning in NLP **#arxiv2019** -- `Nice paper`
* Intriguing properties of randomly weighted networks: Generalizing while learning next to nothing **#crv2019** -- `Nice paper`
* Semi-supervised Domain Adaptation via Minimax Entropy
	> Interesting `PI_Reading_Grp`
* The lottery ticket hypothesis: Finding sparse, trainable neural networks **#iclr2019**
* Imagenet-Trained Cnns Are Biased Towards Texture; Increasing Shape Bias Improves Accuracy And Robustness **#iclr2019**
* Updates in Human-AI Teams: Understanding and Addressing the Performance/Compatibility Tradeoff `AAAI2019` 
	> Well motivated paper!
* Contrastive Multiview Coding **#arxiv2019**
* Recurrent independent mechanisms **#arxiv2019**
* A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms **#arxiv2019**
	> Heavy reading but it is worth it
* Efficient and Effective Dropout for Deep Convolutional Neural Networks **#arxiv2019**
* Visualizing deep similarity networks **#wacv2019**
* Spreading vectors for similarity search `ICLR2019`
* Proxylessnas: Direct Neural Architecture Search On Target Task And Hardware `ICLR2019`
* Dynamic channel pruning: Feature boosting and suppression `ICLR2019`
* Learning deep representations by mutual information estimation and maximization `ICLR2019`
* Slimmable Neural Networks `ICLR2019` `Nice`
	> Train neural network with different widths (switches) [1.0,0.5]. The key idea to use different BatchNorm layers for different switches. Slimmable networks brings multiple advantages: (1) Train a single network compared to multiple individual networks, (2) adjust the computational cost budget of a network during inference on the fly, (3) avoid downloading individual models to the deployment environment.
* Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation **#icml2019**
* EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks **#icml2019** `Arch_Design`
* Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask `NIPS2019`
* This Looks Like That: Deep Learning for Interpretable Image Recognition `NIPS2019`
	> DL Interpretability paper. Learn m=10 prototypes per class. ProtoPNet goes through three stages. Stage 1: Train all layers including prototypes while fixed the weights of the final linear classifier layer; the linear classifier is _not_ randomly initialized but manually initialized to strengthen the connection between prototype p_k and class k, and weaken the connection between prototype p_k and class c!=k. Stage 2: project the learned prototype p_k to the nearest training image patches. Stage 3: tune the linear classifier layer while imposing a lasso regularizer.  
* Lookahead Optimizer: k steps forward, 1 step back `NIPS2019` `Nice`
	> take k steps in optimizer direction, then weight-average the start and end points for a new weight initialization. Repeat; seems promising so far. Can be easily integrated on top of other optimizers (SGD, Adam).
* Transfusion: Understanding Transfer Learning for Medical Imaging `NIPS2019`
	> Nice paper
* Fixing the train-test resolution discrepancy `NIPS2019`
* Stand-Alone Self-Attention in Vision Models `NIPS2019`
* Root Mean Square Layer Normalization `NIPS2019` `UK`
	> Propose an alternative layernorm layer that skip the centering operation. The new layernorm (RMSNorm) only perform scaling operation. It forces the inputs into a sqrt(n)-scaled unit sphere. The paper shows that the scaling operation is more vital compared to centering operation.
* MixMatch: A Holistic Approach to Semi-Supervised Learning `NIPS2019` `Google`
	> Mixup + Semi-Supervised learning
* Consistency-based Semi-supervised Learning for Object Detection `NIPS2019`
	> Nice paper
* SketchEmbedNet: Learning Novel Concepts by Imitating Drawings `NIPS2019`
* Adversarial Training for Free `NIPS2019`
* One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers `NIPS2019`
* Channel Gating Neural Networks `NIPS2019`
* When does label smoothing help? `NIPS2019`
* Hidden stratification causes clinically meaningful failures in machine learning for medical imaging `NIPS2019` `ML4H` `Abstract`
	> Hidden stratification exists in medical imaging. These can be dedicated through one of the following approaches (1) [full-manual] Schema Completion, (2) [partial-manual] Error Auditing, (3) Unsupervised Clustering
* Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks `NIPS2019`
* Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs **#nipsWorkshop2019**
* TriResNet: A Deep Triple-stream Residual Network for Histopathology Grading **#iciar2019**
* Incremental learning in deep convolutional neural networks using partial network sharing **IEEEAccess2019**
* Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks `NeuroComputing2019`
	> test-time augmentation for aleatoric uncertainty estimation for medical image segmentation.
* The NYU Breast Cancer Screening Dataset v1.0 `NYU` **#2019**
* Screening Mammogram Classification with Prior Exams `NYU` **#midl2019**
* Deep Neural Networks Improve Radiologists’ Performance in Breast Cancer Screening `NYU` **#IEEE Medical Imaging 2019**
* Can we reduce the workload of mammographic screening by automatic identification of normal exams with artificial intelligence? A feasibility study `EU` `Rad` `2019`
	> pre-select exams for rads; reduce workload by 17% while missing 1% True positives; threshold tuned on an independent data sample. As a rule of thumb, a population has 7 cancers per 1000; 5 caught, 2 missed.
* Explainable artificial intelligence for breast cancer: A visual case-based reasoning approach `AI` `medicine` `Nice`
	> The paper proposed two visualization approaches for case retrieval in medical images. The two approaches are : (1) polar scatter plots, (2) rainbow box. The paper ran a clinicial study with 11 participants to get feedback on their proposal.
* Improving Language Understanding by Generative Pre-Training **2018** -- `GPT-1` `OpenAI`
* Language Models are Unsupervised Multitask Learners **2019** -- `GPT-2` `OpenAI`
* Deep Image Prior **#bayes #cvpr2018**
* CondenseNet: An Efficient DenseNet Using Learned Group Convolutions **#cvpr2018**
* Neural baby talk **#cvpr2018**
* Blockdrop: Dynamic inference paths in residual networks **#cvpr2018**
* Hydranets: Specialized dynamic architectures for efficient inference **#cvpr2018**
* NISP: Pruning Networks Using Neuron Importance Score Propagation **#cvpr2018**
* Interpret neural networks by identifying critical data routing paths **#cvpr2018**
* A Style-Based Generator Architecture for Generative Adversarial Networks **#cvpr2018**
* Improvements to context based self-supervised learning **#cvpr2018**
* Large-scale distance metric learning with uncertainty **#cvpr2018**
* “Zero-Shot” Super-Resolution using Deep Internal Learning **#cvpr2018**
* Joint optimization framework for learning with noisy labels **#cvpr2018**
* Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination **#cvpr2018** -- interesting 
* Squeeze-and-Excitation Networks **#cvpr2018**
* Data Distillation: Towards Omni-Supervised Learning **#cvpr2018**
* Between-class learning for image classification **#cvpr2018**
* MobileNetV2: Inverted Residuals and Linear Bottlenecks **#cvpr2018**
* Boosting Self-Supervised Learning via Knowledge Transfer **#cvpr2018**
* Separating Style and Content for Generalized Style Transfer **#cvpr2018**
* Local descriptors optimized for average precision **#cvpr2018**
* Cleannet: Transfer learning for scalable image classifier training with label noise. **#cvpr2018**
* Mining on Manifolds: Metric Learning without Labels **#cvpr2018**
* Weakly supervised instance segmentation using class peak response -- **cvpr2018** important :hash:code
* Dynamic Deep Neural Networks: Optimizing Accuracy-Efficiency Trade-offs by Selective Execution **aaai2018**
* VSE++: Improved visual-semantic embeddings. **#bmvc2018#**
* Rise: Randomized input sampling for explanation of black-box models  **#bmvc2018#**
* Compositing-aware image search `ECCV2018` :hash:industry :hash:adobe 
* Progressive neural architecture search `ECCV2018` :hash:industry :hash:Google 
* ConvNets and ImageNet Beyond Accuracy: Understanding Mistakes and Uncovering Biases `ECCV2018`
* Hierarchy of Alternating Specialist for Scene Recognition `ECCV2018`
* Bisenet: Bilateral Segmentation Network For Real-Time Semantic Segmentation `ECCV2018`
* CBAM: Convolutional Block Attention Module `ECCV2018`
* Exploring the Limits of Weakly Supervised Pretraining `ECCV2018`
* SkipNet: Learning Dynamic Routing in Convolutional Networks `ECCV2018`
* Deep metric learning with hierarchical triplet loss `ECCV2018`
* Deep Clustering for Unsupervised Learning of Visual Features -- Nice Paper `ECCV2018`
* Compositional Learning for Human Object Interaction `ECCV2018` -- `PI_Reading_Grp`
* Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks `WACV2018`
* Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks `MIDL2018`
	> Use test-time augmentation for aleatoric uncertainty estimation with application in medical field.
* Self-Attention with Relative Position Representations **#naacl2018**
* Sanity checks for saliency maps. **#nips2018**
* Neighbourhood Consensus Networks. **#nips2018**
* Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels **#nips2018**
	> 	Assumes a noisy dataset; propose a loss function that assigns low weights to noisy (hard) samples and high weights to correct (easy) samples.
* Visualizing the Loss Landscape of Neural Nets **#nips2018** `Nice` `UMD`
	> Run approx 2500 evaluation on the validation set; accordingly better use mpi
* Mixup: Beyond Empirical Risk Minimization **#iclr2018**
* Meta-learning for semi-supervised few-shot classification **#iclr2018**
* Progressive growing of gans for improvedquality, stability, and variation **#iclr2018**
* Born again neural networks. **#icml2018**
* Mine: mutual information neural estimation **#icml2018**
* Similarity of Neural Network Representations Revisited **#icml2018**
* Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. **#icml2018**
* Fast decoding in sequence models using discrete latent variables **#icml2018** -- Nice paper
* Stochastic video generation with a learned prior **#pmlr2018** -- `PI_Reading_Grp`
* Accelerating deep metric learning via cross sample similarities transfer ([code](https://github.com/TuSimple/DarkRank/blob/master/PYOP/listmle_loss.py)) **#aaai2018**
* Understanding Deep Convolutional Networks through Gestalt Theory **#ist2018**
* Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation **#corl2018**
* Semantic Segmentation from Limited Training Data **#icra2018**
* Representation Learning with Contrastive Predictive Coding **#arxiv2018**
* Dataset Distillation -- `BAIR` **#arxiv2018**
* fastMRI: An Open Dataset and Benchmarks for Accelerated MRI -- `NYU` **#arxiv2018**
* Learning a Variational Network for Reconstruction of Accelerated MRI Data -- `NYU` **Magnetic Resonance In Medicine**
* Towards End-to-End Lane Detection: an Instance Segmentation Approach **IEEE intelligent vehicles symposium 2018**
* SpectralNet: Spectral Clustering using Deep Neural Networks **#arxiv2018**
* The Singular Values of Convolutional Layers **#arxiv2018**
* Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer **#arxiv2018**
* Stochastic Adversarial Video Prediction **#arxiv2018**
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding **#arxiv2018**
* Large scale distributed neural network training through online distillation **#arxiv2018**
* Learning Low-Rank Representations **#arxiv2018**
* Label refinery: Improving imagenet classification through label progression **#arxiv2018**
* Stabilizing Gradients for Deep Neural Networks via Efficient SVDParameterization **#arxiv2018**
* Flexible deep neural network processing **#arxiv2018**
* Deep Paper Gestalt **#arxiv2018**
* Top-down neural attention by excitation backprop **#ijcv2018**
* MonolithNet: Training monolithic deep neural networks via a partitioned training strategy **#jcvis2018**
* Detecting and classifying lesions in mammograms with Deep Learning **#nature2018**
* DeepLesion: Automated Deep Mining, Categorization and Detection of Significant Radiology Image Findings using Large-Scale Clinical Lesion Annotations **#nlm2018**
* GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification **Neurocomputing** **2018**
* Amazon Inventory Reconciliation Using AI `Github` `AMZN` `Dataset`
* A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning **#distill #cvpr2017**
* ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases **#cvpr2017**
* From Red Wine to Red Tomato: Composition with Context **#cvpr2017**
* Focal Loss for Dense Object Detection **#iccv2017**
* Online Video Object Detection using Association LSTM **#iccv2017**
	> Seems interesting but no code released!

* SVDNet for Pedestrian Retrieval **#iccv2017**
* Revisiting unreasonable effectiveness of data in deep learning era **#iccv2017**
* Arbitrary style transfer in real-time with adaptive instance normalization. **#iccv2017**
* Learning efficient convolutional networks through network slimming. **#iccv2017**
* Deep Metric Learning with Angular Loss **#iccv2017**
* Representation Learning by Learning to Count **#iccv2017**
* Dynamic coattention networks for question answering **#coattention** `ICLR2017`
* Adversarial feature learning `ICLR2017`
* learned representation for artistic style. `ICLR2017` 
* beta-vae: Learning basic visual concepts with a constrained variational framework `ICLR2017`
* A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks `ICLR2017`
* Sgdr: Stochastic gradient descent with warm restart `ICLR2017`
* Density Estimation using Real-NVP `Canada` `Mila` `Google` `ICLR2017`
	> Propose a flow-based generative model that is non-volume preserving with efficient Jacobian determinant computation. The paper propose affine-coupling layers that are invertible. The paper propose and leverage multi-scale and BatchNorm within their layers for better expressivity
* Split-brain autoencoders: Unsupervised learning by cross-channel prediction **cvpr2017**
* Universal Adversarial Perturbations **cvpr2017**
* Feature Pyramid Networks for Object Detection **cvpr2017**
* YOLO9000: Better, Faster, Stronger **cvpr2017**
* spatially adaptive computation time for residual networks **cvpr2017**
* No fuss distance metric learning using proxies **cvpr2017**
* iCaRL: Incremental classifier and representation learning **cvpr2017**
* Deep Mutual Learning **#nips2017**
* Runtime neural pruning **#nips2017**
* Train longer, generalize better: closing the generalization gap in large batch training of neural networks **#nips2017**
* Dynamic Routing Between Capsules **#nips2017**
* Dual Discriminator Generative Adversarial Nets **#nips2017**
* VEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning **#nips2017**
* Dual Path Networks **#nips2017**
* The Reversible Residual Network: Backpropagation Without Storing Activations **#nips2017**
* Active Bias: Training More Accurate Neural Networks by Emphasizing High Variance Samples **#nips2017**
* Learning Spread-out Local Feature Descriptors **#iccv2017**
* Thinet: A filter level pruning method for deep neural network compression. **#iccv2017**
* Channel pruning for accelerating very deep neural networks. **#iccv2017**
* Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. **#iccv2017**
* Drone-based Object Counting by Spatially Regularized Regional Proposal Networks **#iccv2017**
* Learning Efficient Convolutional Networks through Network Slimming **#iccv2017**
* Demystifying Neural Style Transfer **#ijcai2017**
* Working hard to know your neighbor’s margins: Local descriptor learning loss  -- Triplet Lifted Structure loss **nips2017**
* TRACE NORM REGULARIZATION AND FASTER INFER-ENCE FOR EMBEDDED SPEECH RECOGNITION RNNS **#arxiv2017**
* SmoothGrad: removing noise by adding noise **#arxiv2017**
* EmotioNet Challenge: Recognition of facial expressions of emotion in the wild **#arxiv2017**
	> Motivation?
* Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour **#arxiv2017**
* High-Resolution Breast Cancer Screening with Multi-View Deep Convolutional Neural Networks -- `WR-AI` **#arxiv2017**
* Large Batch Training of Convolutional Networks **#arxiv2017** `Nvidia`
	> Nice and simple
* Neural Discrete Representation Learning **#arxiv2017**
* MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications **#arxiv2017**
* Unsupervised learning by predicting noise. **#icml2017**
* A Closer Look at Memorization in Deep Networks **#icml2017** `Mila`
	> Nice analysis
* Learning to Generate Long-term Future via Hierarchical Prediction **#icml2017**
* SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization **#icml2017**
* Prototypical networks for few-shot learning **#nips2017**
* Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results **#nips2017**
* A closer look at memorization in deep networks -- Well written paper **#icml2017**
* DSD: Dense-Sparse-Dense Training for Deep Neural Networks **#iclr2017**
* Understanding deep learning requires rethinking generalization **#iclr2017**
	> Regularizatin is not enough to explain generalization. Most networks can easily over-fit (zero-train loss) on large datasets even when images are assigned random labels.
* Temporal ensembling for semisupervised learning **#iclr2017** -- Dropout Potential
* Pruning filters for efficient convnets **#iclr2017**
* Squeezenet: Alexnet-Level Accuracy With 50X Fewer Parameters And <0.5Mb Model Size **#iclr2017**
* Membership Inference Attacks against Machine Learning Models
	> Overfitting models suffer membership attacks. Regularization helps mitigate this problem.

* Rethinking Atrous Convolution for Semantic Image Segmentation `arXiv2017` `Arch_Design`
* Emotion Detection Using Noninvasive Low Cost Sensors `IT` `ACII2017`
	> Use non-invasive sensors to classify emotions. The senors used are EEG for Brain waves, EMG for muscle contraction, and GSR for skin conductance. EMG seems to be less effective, but the sensor was attached to the participant arms. Classic classifier models (e.g., SVMs) are used. The dataset is, of course, tiny by today standards.
* Learning local image descriptors with deep siamese and triplet convolutional networks by minimising global loss functions **#cvpr2016**
* Sketch Me That Shoe **#cvpr2016**
* Convolutional Pose Machines **#cvpr2016**
* You Only Look Once: Unified, Real-Time Object Detection **#cvpr2016**
* Embedding Label Structures for Fine-Grained Feature Representation **#cvpr2016**
* Learning local image descriptors with deep siamese and triplet convolutional networks by minimising global loss functions **#cvpr2016**
* Large Scale Semi-supervised Object Detection using Visual and Semantic Knowledge Transfer **#cvpr2016**
* Factors in finetuning deep model for object detection with long-tail distribution **#cvpr2016**
* Joint unsupervised learning of deep representations and image clusters **#cvpr2016**
* Unsupervised learning of visual representations by solving jigsaw puzzles **#eccv2016**
* Colorful Image Colorization **#eccv2016**
* Learning without Forgetting **#eccv2016**
* Identity Mappings in Deep Residual Networks **#eccv2016**
* Less Is More: Towards Compact CNNs **#eccv2016**
* Ask Me Anything:
Dynamic Memory Networks for Natural Language Processing **#icml2016**
* Unsupervised Deep Embedding for Clustering Analysis **#icml2016**
* Dynamic Capacity Network **#icml2016**
* Wide Residual Networks **#bmvc2016**
* Context Matters: Refining Object Detection in Video with Recurrent Neural Networks **#bmvc2016**
	> Feeding B BBox (and their features) to GRU
* Adaptive convolutional neural network and its application in face recognition **NeuralProcessingLetters2016**
* Improved Deep Metric Learning withMulti-class N-pair Loss Objective **#nips2016** 
* Understanding the Effective Receptive Field in Deep Convolutional Neural Networks **#nips2016**  -- Nice
* Residual Networks Behave Like Ensembles of Relatively Shallow Networks **#nips2016** 
* Improved techniques for training GANs **#nips2016** 
* Dynamic Filter Networks **#nips2016** 
* Universal Correspondence Network **#nips2016** 
* Learning Deep Parsimonious Representations **#nips2016** 
* InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets **#nips2016** 
* PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions **#nips2016** 
* Conditional image generation with pixelcnn decoders **#nips2016** 
* Convolutional neural networks with low-rank regularization **#iclr2016** -- [code](https://github.com/chengtaipu/lowrankcnn)
* All you need is a good init **#iclr2016**
* Reducing Overfitting In Deep Networks By Decorrelating Representations **#iclr2016**
* Unifying distillation and privileged information **#iclr2016**
* Adversarially Learned Inference **#arxiv2016** 
* ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation **#arxiv2016** `Arch_Design`
* Gaussian Error Linear Units (GELUs) **#arxiv2016** 
* Pruning Filters For Efficient Convnets **#arxiv2016** 
* Understanding deep learning requires rethinking generalization **#arxiv2016**
* Understanding intermediate layers using linear classifier probes **#arxiv2016** 
* What makes ImageNet good for transfer learning? **#arxiv2016** 
* Adaptive Computation Time for Recurrent Neural Networks **#arxiv2016** 
* Unrolled generative adversarial networks **#iclr2016** 
* Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization And Huffman Coding **#iclr2016** 
* Data-dependent initializations of convolutional neural networks **#iclr2016** 
* Learning deep embeddings with histogram loss **#nips2016**
* Hierarchical question-image co-attention for visual question answering. **#coattention** **#nips2016**
* What’s the point: Semantic segmentation with point supervision **#eccv2016**
* Deep networks with stochastic depth **#eccv2016**
* Particular object retrieval with integral max-pooling of cnn activations **#iclr2016**
* Bayesian representation learning with oracle constraints. **#iclr2016**
* The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies -- **#acm2016**
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks **#tpami2016** -- `Anchors` and `RPN` origins
* "Why Should I Trust You?": Explaining the Predictions of Any Classifier `SIGKDD2016`
	> LIME for explaining predictions. Approximate decision boundary in a local neighborhood
* A Similarity Study of Interactive Content-Based Image Retrieval Scheme for Classification of Breast Lesions `FDA` `Trans Information and Systems`
	> Interactive query expansion for medical case retrieval using Rocchio relevance feedback (RRF) algorithm. No clinical studies; just evaluation for the impact of RRF. Seems to be a paper revision
* Interactive Content-Based Image Retrieval (CBIR) Computer-Aided Diagnosis (CADx) System for Ultrasound Breast Masses using Relevance Feedback `MedicalImaging2011`
	> Interactive query expansion for medical case retrieval using Rocchio relevance feedback (RRF) algorithm. No clinical studies; just evaluation for the impact of RRF. 
* Learning to compare image patches via convolutional neural networks **#cvpr2015**
* Matchnet: Unifying feature and metric learning for patch-based matching **#cvpr2015** **#industry** 
* Understanding image representations by measuring their equivariance and equivalence **#cvpr2015** `Oxford`
	> equivariance: A single model applied on two image variants should generate two representations that are equivalent through a matrix transformation. Equivalence: Two models applied on a single image should generate the similar/replacable representations such that the two models can be stitched on top of each other. 
* FaceNet: A unified embedding for face recognition and clustering **#cvpr2015** **#industry** 
* Efficient object localization using convolutional networks **#cvpr2015** 
* Spatial Transformer Network **#nips2015**
* Deep visual analogy-making **#nips2015**
* Wide-area image geolocalization with aerial reference imagery **#iccv2015**
* Fast R-CNN **#iccv2015**
* Delving deep into rectifiers: Surpassing human-level performance on imagenet classification **#iccv2015**
* Discriminative unsupervised feature learning with exemplar convolutional neural networks **#pami2015**
* Learning both Weights and Connections for Efficient Neural Networks `ICLR2015`
* Fitnets: Hints for thin deep nets `ICLR2015`
* Flattened convolutional neural networks for feedforward acceleration `ICLR2015` `Workshop`
* NICE: Non-Linear Independent Components Estimation `ICLR2015`
* Training Deep Neural Networks on Noisy Labels with Bootstrapping `ICLR2015` `Workshop`
* Understanding Locally Competitive Networks `ICLR2015`
* Deeply-Supervised Nets `PMLR2015`
	> Apply loss on intermediate features so it is more discriminative
* Highway networks  `arXiv2015`
* Multi-path Convolutional Neural Networks for Complex Image Classification `arXiv2015`
	> very primitive but interesting
* Learning fine-grained image similarity with deep ranking **#cvpr2014**
* DeepPose: Human Pose Estimation via Deep Neural Networks **#cvpr2014**
* Rich Feature Hierarchies For Accurate Object Detection And Semantic Segmentation **#cvpr2014** `R-CNN`
* LSDA: Large Scale Detection Through Adaptation **#nips2014**
* Deep convolutional network cascade for facial point detection **#cvpr2013**
* Write a Classifier: Zero-Shot Learning Using Purely Textual Description **#iccv2013**
	> Heavy numerical optimization
* Fast dropout training **#icml2013**
* Adaptive dropout for training deep neural networks **#nips2013**
* Maxout networks **#arvix2013** -- When Goodfellow was young
* Clinical Experience Sharing by Similar Case Retrieval `Turkey` `ACM` `Workshop` `MIIRH2013`
	> Create a retrieval-based system for 3D liver volumes. The system uses both image and non-image modality. The paper motivates the system for clinical experience sharing. This can be utilized in for educational purposes and preventing "situations, where lack of medical experience might have negative effects on diagnosis". Unfortunately, the proposed system is never evaluated in a clinical settings. The proposal is evaluated only using a toy retrieval setup.
* AVA: A Large-Scale Database for Aesthetic Visual Analysis **#cvpr2012**
* Measuring the objectness of image windows **#pami2012**
* INbreast: Toward a Full-field Digital Mammographic Database **#nlm2012**
* Unbiased Look at Dataset Bias **#cvpr2011**
* The Power of Comparative Reasoning -- `PI_Reading_Grp` **#iccv2011**
* On random weights and unsupervised feature learning **#icml2011**
* Understanding the difficulty of training deep feedforward neural networks **#ai & stats 2010**
* Why does unsupervised pre-training help deep learning **#jmlr2010**
* What is the best multi-stage architecture for object recognition? **#iccv2009**
* Measuring Invariances in Deep Networks `NIPS2009`
* Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning `NIPS2008`
* Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval `ICCV2007`
* How to Read a Paper `2007`
	> First pass 5 mins, second pass 1 hr, third pass 4-5 hrs for beginners/1 hr for experienced
* Greedy Layer-Wise Training of Deep Networks `NIPS2006`
* The development of embodied cognition: Six lessons from babies. **Artificiallife2005**
* Semi-supervised learning by entropy minimization `NIPS2004` -- A simple and good idea but the notation! 
* The Dynamic Representation of Scenes **#VisCog2000**
* Separating Style and Content **#nips1997**
* Training feedforward neural networks using genetic algorithms **#IJCAI1989**
* Simplifying neural networks by soft weight-sharing **#Neural computation 1992** -- I admire Hinton's writing style
* Distributed Representation Geoffrey Hinton -- heavy reading but worth it **CMU 1984**
* K-Lines: A Theory of Memory **Cognitive Science 1980**
	> I like it. Yet, it felt more like a brain teaser than an academic paper.
	
	