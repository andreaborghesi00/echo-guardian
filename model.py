# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detection of breast cancers
# Train model on CT and PET scans to predict malignant or benign cancer by validating it with histological positive or negative labels
# - Radiologist accuracy: 80% -> Get an accuracy higher than 80%
# ### What to do
# - Anatomical location of the region
# - Characterization of the region
# - Therapeutic choice
# - Monitoring of the therapy to monitor patient's response to treatment
# ### Risk Assessment Analysis -> High Risk tools
# Include Risk Assessment Analysis of the tool in pitch deck
# - Include list of requirements of our system: operating system, tools, forms, etc, cybersecurity risks (login)
# - Tool performances in terms of
#     - Sensitivity (TP rate) -> **at least 90%**
#     - FP rate
#     - Time
# ### Model testing during presentation
# **1 second as maximum response time for testing** -> model will be tested live during presentation
# ### Data privacy
# - Require auth
# - No personal info (name, surname, TAX ID code, address, etc)
# - Cluster ages to reduce privacy risk (each 5 years?)
# ### Requirements
# - DLL library to call functions from -> **segregation of functions**
# - Must have **EXIT** button
# - Cybersecurity of the tool
# - AUC greather than 0.85
# - Sensitivity higher than 0.90
#
# ### Ethics -> **INCLUDE IT**
# Simulate that we actually got these documents in the first few slides
# - Informed consent (got from the patient)
# - Ethic Committee Approval (about 2 months after study request)
#     - It was approved
# - Anonimization
#     - Define the measures in the data
#
# ### Methods to be used:
# - Radiomics library
# - Random Forest or SVM (choose one, maybe PCA to choose the best features)
# - CNNs
#
# ### Dataset (with masks)
# - 437 benign
# - 210 malignant

# %%

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import radiomics
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from RadiomicsDataset import RadiomicsDataset
import torchmetrics
from torchsummary import summary
from tqdm import tqdm

# %% [markdown]
# # Data loading

# %%
path = 'dataset/'
df = pd.DataFrame(columns=['image', 'mask', 'label'])

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        if file.find('mask') == -1:
            df = pd.concat([df, pd.DataFrame({'image': [os.path.join(path, folder, file)],
                                        'mask': [os.path.join(path, folder, file.replace('.png', '_mask.png'))],
                                        'label': [0 if 'benign' in file else 1]})])
df.index = range(1, len(df) + 1)
df

# %% [markdown]
# ### Sample: Radiomics feature extraction

# %%
glcm_feats = keys_list = [
    # 'original_glcm_Autocorrelation',
    'Autocorrelation',
    'ClusterProminence',
    'ClusterShade',
    'ClusterTendency',
    'Contrast',
    'Correlation',
    'DifferenceAverage',
    'DifferenceEntropy',
    'DifferenceVariance',
    'Id',
    'Idm',
    'Idmn',
    'Idn',
    'Imc1',
    'Imc2',
    'InverseVariance',
    'JointAverage',
    'JointEnergy',
    'JointEntropy',
    'MCC',
    'MaximumProbability',
    # 'SumAverage',
    'SumEntropy',
    'SumSquares'
]

# %%
# load one image and mask as a sample as numpy array
image_path = df['image'][2]
mask_path = df['mask'][2]

# Configure the feature extractor
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
# extractor.enableAllFeatures()
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName('firstorder')

extractor.enableAllFeatures()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('shape2D')
extractor.enableFeaturesByName(glcm=glcm_feats)
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')
# extractor.enableFeatureClassByName()

image = sitk.ReadImage(image_path, sitk.sitkInt32)
mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
features = extractor.execute(image, mask, voxelBased=False, label=255)

# %%
clean_features_dict = {key: float(features[key]) for key in features if key.startswith('original_')}
clean_features_dict # these are they keys features that we will work with

# %%
len(clean_features_dict)

# %% [markdown]
# ## Splitting, datasets, data loaders

# %%
img_mask_paths = [(df['image'].iloc[idx], df['mask'].iloc[idx]) for idx in range(len(df))]
labels = df['label'].values

# %%
data_train, data_tmp, label_train, label_tmp = train_test_split(img_mask_paths, labels, test_size=0.2, random_state=69420, stratify=labels)
data_val, data_test, label_val, label_test = train_test_split(data_tmp, label_tmp, test_size=0.5, random_state=69420, stratify=label_tmp)

# %%
train_ds = RadiomicsDataset(data_train, label_train)
val_ds = RadiomicsDataset(data_val, label_val)

# %%
len(train_ds.__getitem__(0)[0])

# %%
batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size)

# %% [markdown]
# # Classifiers

# %% [markdown]
# ## Neural Networks

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [markdown]
# ### Fully-Connected only

# %%
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(101, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# %%
simple_net = SimpleNet().to(device)


# %%
def validate(model, data_loader):
    macro_acc_metric = torchmetrics.Accuracy(task='binary', average='macro')
    micro_acc_metric = torchmetrics.Accuracy(task='binary', average='micro')
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            macro_acc_metric.update(output, target)
            micro_acc_metric.update(output, target)
    return macro_acc_metric.compute(), micro_acc_metric.compute()


# %%
def train(model, train_loader, val_loader, optimizer, loss_criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            
            loss = loss_criterion(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        macro_acc, micro_acc = validate(model, val_loader)
        print(f'Epoch: {epoch}, Macro Acc: {macro_acc}, Micro Acc: {micro_acc}')


# %%
optimizer = optim.Adam(simple_net.parameters(), lr=0.001)
loss_criterion = nn.CrossEntropyLoss()

train(simple_net, train_dl, val_dl, optimizer, loss_criterion, epochs=10)

# %% [GUI]
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import SimpleITK as sitk

print(df.iloc[0, 0])
img = sitk.ReadImage(df.iloc[0, 0], sitk.sitkInt32)
np.shape(img)
fig = px.imshow(img)

fig.update_layout(
    dragmode="drawopenpath",
    newshape_line_color="cyan",
    title_text="Draw a path to separate versicolor and virginica",
)
config = dict(
    {
        "scrollZoom": True,
        "displayModeBar": True,
        # 'editable'              : True,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
        "toImageButtonOptions": {"format": "svg"},
    }
)

st.plotly_chart(fig, config=config)

# %%
