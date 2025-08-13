Simple Deepfake Detection Baselines using Vision Transformers (ViT), Swin Transformers, and Efficientnet

These are different backbone models attached with a binary classifier to find the most appropriate backbone.  
These are purely for baseline, with no further changes

All models are trained on DFDC sample dataset available for free on Kaggle, and a laptop with RTX 3050 6 GB  
MTCNN is used for Face Extraction, and Extracted faces are stored in a local folder

ViT_FeatEx - Feature extraction using ViT then training classifier on the features  
ViT_FC - Attaching classifier on top of ViT and training it with last few layers of ViT  
ViT_FC_comb - combining ViT and Classifier in a single class/model

These models are trained similarly to ViT_FC_comb  
SwinT - Swin Transformer Tiny  
SwinT_base - Swin Transformer Base  
EfficientNet_B0 - EfficientNet B0



All open source content used belongs to their respective owners.
