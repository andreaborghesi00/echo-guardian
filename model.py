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
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import radiomics
import imageio.v2 as iio
import SimpleITK as sitk

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

# %%
# load one image and mask as a sample as numpy array
image_path = df['image'][2]
mask_path = df['mask'][2]


# %%
# from radiomics import featureextractor

# Configure the feature extractor
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName('firstorder')
extractor.disableFeatureByName


# %%
image = sitk.ReadImage(image_path, sitk.sitkInt32)
mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
features = extractor.execute(image, mask, voxelBased=False, label=255)

# %%
import six
for key, val in features.items():
    print("\t%s: %s" %(key, val))
len(features)

