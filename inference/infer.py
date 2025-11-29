from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from transformers import logging
logging.set_verbosity_error()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

base = r"D:\W\VS\VS Folder\DFD\DFD-T"
model_path = rf'{base}\swin_tiny.pth'

valid_train_df = pd.read_csv(rf"{base}\valid_train_df.csv")



# Analysis of values
label_counts = valid_train_df['label'].value_counts()
label_0_count = label_counts.get(0, 0)
label_1_count = label_counts.get(1, 0)
print("\n--------------Label distribution in valid_train_df------------------")
print(f"Label 0 count: {label_0_count}, Label 1 count: {label_1_count}")



# Select a specific row for inference
row=valid_train_df.iloc[330] # Example 330 for 1, 192 for 0
print("\n--------------Selected video and class------------------")
print(os.path.basename(row['path']),',', row['label'])
path= row['path']
label = row['label']



# Load the model and processor
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=1, ignore_mismatched_sizes=True)
criterion = nn.BCEWithLogitsLoss(reduction='none')  # For binary classification



# Load weights
model.load_state_dict(torch.load(model_path))
model = model.to(device)



# Inference
batch_size = 36
pred_classes = []
total_batches = 0
losses = []
probabs = []
correct = 0
total = 0
total_batches = 0

image_files = sorted([ os.path.join(path, f)
for f in os.listdir(path)
if f.lower().endswith('.png')
])

print("\n--------------Running Inference------------------")
model.eval()
for i in tqdm(range(0, len(image_files), batch_size), desc='Image Batches'):
    total_batches += 1
    batch_paths = image_files[i:i+batch_size]
    images = [Image.open(p).convert("RGB") for p in batch_paths]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    batch_labels = torch.tensor([label] * len(images), dtype=torch.float).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        batch_logits = outputs.logits

        # running_loss += criterion(batch_logits, batch_labels).item()* batch_labels.size(0)
        loss = criterion(batch_logits, batch_labels).detach().cpu().view(-1).tolist()
        losses.extend(loss)

        prob=torch.sigmoid(batch_logits).detach().cpu().view(-1).tolist()
        probabs.extend(prob)

        pred = (torch.sigmoid(batch_logits) > 0.5).int()
        correct += (pred == batch_labels.int()).sum().item()
        total += batch_labels.size(0)
        pred_classes.extend(pred.cpu().numpy().flatten().tolist())
        

accuracy = correct / total if total > 0 else 0
val_loss = sum(losses) / total if total > 0 else 0

pred = int(sum(pred_classes) > len(pred_classes) / 2)
print("\n--------------Result------------------")
print(f"Predicted class: {pred}, Actual class: {label}")

