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
from SegmentationDataset import SegmentationDataset
import albumentations as A
import torchmetrics
from torchsummary import summary
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torchvision import transforms

# %% [markdown]
# # Data loading

# %%
path = 'dataset/'

df = pd.DataFrame(columns=['image', 'mask', 'label'])

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        # Only loop through the masks so that we can handle duplicate masks easily
        if 'mask' in file:
            img_file = file.split('_mask')[0] + '.png'
            df = pd.concat([df, pd.DataFrame({
                'image': [os.path.join(path, folder, img_file)],
                'mask': [os.path.join(path, folder, file)],
                'label': [1 if 'malignant' in file else 0]
            })])

df.index = range(1, len(df) + 1)
print(df.shape)
print(df.head(25))

# %% [markdown]
# ## Splitting, datasets, data loaders

# %%
img_mask_paths = [(df['image'].iloc[idx], df['mask'].iloc[idx]) for idx in range(len(df))]
labels = df['label'].values
print(np.shape(labels))
print(labels)

# %%
data_train, data_valtest, label_train, label_valtest = train_test_split(img_mask_paths, labels, test_size=0.2, random_state=69420, stratify=labels)
data_val, data_test, label_val, label_test = train_test_split(data_valtest, label_valtest, test_size=0.5, random_state=69420, stratify=label_valtest)

# %%
# augmentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    # A.RandomGamma(p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
])

val_transform = A.Compose([
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    A.GaussNoise(p=0.5),
])


# %%
# Create and save or load the dataloaders
batch_size = 64
force_dataset_creation = False
data_dir = 'data'

train_classifier_ds_path = os.path.join(data_dir, 'train_classifier_ds.pt')
test_classifier_ds_path = os.path.join(data_dir, 'test_classifier_ds.pt')
val_classifier_ds_path = os.path.join(data_dir, 'val_classifier_ds.pt')

if (os.path.exists(train_classifier_ds_path) and os.path.exists(val_classifier_ds_path) and os.path.exists(test_classifier_ds_path)) and not force_dataset_creation:
    # Load
    train_classifier_ds = torch.load(train_classifier_ds_path)
    test_classifier_ds = torch.load(test_classifier_ds_path)
    val_classifier_ds = torch.load(val_classifier_ds_path)
    print('Loaded train_classifier_ds, val_classifier_ds and test_classifier_ds from file')

else:
    # Create and save
    
    train_classifier_ds = RadiomicsDataset(data_train, label_train, scaler=RobustScaler(), transform=train_transform, json_exclude_path='excludedImages.json', exclusion_class='classifier')
    test_classifier_ds  = RadiomicsDataset(data_test, label_test, scaler=train_classifier_ds.scaler, transform=val_transform, json_exclude_path='excludedImages.json', exclusion_class='classifier')
    val_classifier_ds   = RadiomicsDataset(data_val, label_val, scaler=train_classifier_ds.scaler, transform=val_transform, json_exclude_path='excludedImages.json', exclusion_class='classifier')
    torch.save(train_classifier_ds, train_classifier_ds_path)
    torch.save(test_classifier_ds, test_classifier_ds_path)
    torch.save(val_classifier_ds, val_classifier_ds_path)
    print('Saved train_classifier_ds, val_classifier_ds and test_classifier_ds to file')


train_classifier_dl = DataLoader(train_classifier_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_classifier_dl  = DataLoader(test_classifier_ds, batch_size=batch_size, num_workers=8)
val_classifier_dl   = DataLoader(val_classifier_ds, batch_size=batch_size, num_workers=8)

# %% [markdown]
# ### Visualize the features

# %%
# print('train')
# print(np.shape(data_train))
# print(np.shape(label_train))
# print(np.shape(train_classifier_ds.rad_features))
# print(np.shape(train_classifier_ds.labels))
# print('val')
# print(np.shape(data_val))
# print(np.shape(label_val))
# print(np.shape(val_classifier_ds.rad_features))
# print(np.shape(val_classifier_ds.labels))
# print('test')
# print(np.shape(data_test))
# print(np.shape(label_test))
# print(np.shape(test_classifier_ds.rad_features))
# print(np.shape(test_classifier_ds.labels))

# %% [markdown]
# # Classifiers

# %% [markdown]
# ## Neural Networks

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# %% [markdown]
# ### Fully-Connected only

# %%
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(101, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1) # 2 classes
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.leaky_relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return torch.sigmoid(x)
    
class RadiomicsNet(nn.Module):
    def __init__(self, input_size=101, hidden_size=1024, num_layers=4, dropout_rate=0.5):
        super(RadiomicsNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size // (2**i), hidden_size // (2**(i + 1)) ))
            self.layers.append(nn.BatchNorm1d(hidden_size // (2**(i + 1)) ))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_size // (2**(num_layers - 1)), 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return F.leaky_relu(x)


# %%
simple_net = SimpleNet().to(device)
radiomics_net = RadiomicsNet().to(device)
print(radiomics_net)

# %%
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice
from torchmetrics import Accuracy, ConfusionMatrix

def validate(model, data_loader):
    model.eval()

    macro_acc_metric = Accuracy(task='binary', average='macro').to(device)
    micro_acc_metric = Accuracy(task='binary', average='micro').to(device)
    cm = ConfusionMatrix(task='binary', num_classes=2).to(device)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = torch.round(output)
            
            macro_acc_metric.update(output, target)
            micro_acc_metric.update(output, target)
            cm.update(output, target)
            
    tn, fp, fn, tp = cm.compute().reshape(-1)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f'\nT      Predicted')
    print(f'r      P  |  N')
    print(f'u   P: {tp} | {fn}')
    print(f'e   N: {fp}  | {tn}')

    return macro_acc_metric.compute(), micro_acc_metric.compute(), specificity.item(), sensitivity.item()


# %%
from functools import wraps

def tqdm_decorator(use_tqdm):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if use_tqdm:
                with tqdm(total=kwargs['epochs'], desc='Training', leave=True, unit='epoch') as pbar:
                    kwargs['pbar'] = pbar
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@tqdm_decorator(use_tqdm=False)
def train(model, train_loader, val_loader, optimizer, loss_criterion, epochs=10, pbar=None):
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # target = target.squeeze().long()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()

        macro_acc, micro_acc, specificity, sensitivity = validate(model, val_loader)
        if pbar is not None:
            pbar.set_description(f'Epoch: {epoch + 1} - Macro Acc: {macro_acc:.3f}, Micro Acc: {micro_acc:.3f}, Specificity {specificity:.3f}, Sensitivity {sensitivity:.3f}')
            pbar.update(1)
        else:
            print(f'Epoch: {epoch + 1} - Macro Acc: {macro_acc:.3f}, Micro Acc: {micro_acc:.3f}, Specificity {specificity:.3f}, Sensitivity {sensitivity:.3f}')


# %%
optimizer = optim.Adam(simple_net.parameters(), lr=1e-3, weight_decay=1e-5)
loss_criterion = nn.BCELoss()

train(simple_net, train_classifier_dl, val_classifier_dl, optimizer, loss_criterion, epochs=0)

# %% [markdown]
# # Segmentation

# %%
import cv2

path = 'dataset_unique_masks/'

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        if 'mask_' in file:
            original_mask = file.split('_mask_')[0] + '_mask.png'
            print(f'original_mask = {original_mask}')
            print(f'second mask = {file}')
            orig_mask = cv2.imread(os.path.join(path, folder, original_mask), cv2.IMREAD_GRAYSCALE)
            second_mask = cv2.imread(os.path.join(path, folder, file), cv2.IMREAD_GRAYSCALE)
            new_mask = cv2.bitwise_or(orig_mask, second_mask)
            cv2.imwrite(os.path.join(path, folder, original_mask), new_mask)
            # !rm "{os.path.join(path, folder, file)}"

df_segment = pd.DataFrame(columns=['image', 'mask', 'label'])

for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path, folder)):
        # Only loop through the masks so that we can handle duplicate masks easily
        if 'mask' in file:
            img_file = file.split('_mask')[0] + '.png'
            df_segment = pd.concat([df_segment, pd.DataFrame({
                'image': [os.path.join(path, folder, img_file)],
                'mask': [os.path.join(path, folder, file)],
                'label': [1 if 'malignant' in file else 0]
            })])

df_segment.index = range(1, len(df_segment) + 1)

# %%
img_mask_paths_segment = [(df_segment['image'].iloc[idx], df_segment['mask'].iloc[idx]) for idx in range(len(df_segment))]
labels_segment = df_segment['label'].values
print(np.shape(labels_segment))
print(labels_segment)


data_train, data_valtest, label_train, label_valtest = train_test_split(img_mask_paths_segment, labels_segment, test_size=0.2, random_state=69420, stratify=labels_segment)
data_val, data_test, label_val, label_test = train_test_split(data_valtest, label_valtest, test_size=0.5, random_state=69420, stratify=label_valtest)

# %%
from albumentations.pytorch import ToTensorV2

segmentation_train_augmentation = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.5),
    A.Blur(blur_limit=5, p=0.5),
    A.Normalize(normalization='min_max'),
    ToTensorV2(),
    
])

# pytorch transforms
segmentation_valtest_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(normalization='min_max'),
    ToTensorV2(),
])
    

# %%
# Create and save or load the dataloaders
batch_size = 16
force_dataset_creation = True
data_dir = 'data'

train_segment_ds_path = os.path.join(data_dir, 'train_segment_ds.pt')
test_segment_ds_path = os.path.join(data_dir, 'test_segment_ds.pt')
val_segment_ds_path = os.path.join(data_dir, 'val_segment_ds.pt')

if (os.path.exists(train_segment_ds_path) and os.path.exists(val_segment_ds_path) and os.path.exists(test_segment_ds_path)) and not force_dataset_creation:
    # Load
    train_segment_dl = torch.load(train_segment_ds_path)
    test_segment_dl = torch.load(test_segment_ds_path)
    val_segment_dl = torch.load(val_segment_ds_path)

    print('Loaded train_segment_ds, val_segment_ds and test_segment_ds from file')

else:
    # Create and save
    train_segment_ds = SegmentationDataset(data_train, label_train,
                                    transform=segmentation_train_augmentation,
                                    json_exclude_path='excludedImages.json',
                                    exclusion_class='cnn')
    
    val_segment_ds = SegmentationDataset(data_val, label_val,
                                    transform=segmentation_valtest_transform,
                                    json_exclude_path='excludedImages.json',
                                    exclusion_class='cnn')
    
    test_segment_ds = SegmentationDataset(data_test, label_test,
                                    transform=segmentation_valtest_transform,
                                    json_exclude_path='excludedImages.json',
                                    exclusion_class='cnn')
    torch.save(train_segment_ds, train_segment_ds_path)
    torch.save(test_segment_ds, test_segment_ds_path)
    torch.save(val_segment_ds, val_segment_ds_path)
    print('Saved train_segment_ds, val_segment_ds and test_segment_ds to file')
    
train_segment_dl = DataLoader(train_segment_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_segment_dl = DataLoader(test_segment_ds, batch_size=batch_size, num_workers=8)
val_segment_dl = DataLoader(val_segment_ds, batch_size=batch_size, num_workers=8)


# %%
def validate_segmentation(model, data_loader):
    iou_metric = BinaryJaccardIndex().to(device)
    dice_metric = Dice(num_classes=1, multiclass=False).to(device)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = torch.sigmoid(output)
            output = torch.round(output)
            output = output.int()
            target = target.int()
            
            iou_metric.update(output, target)
            dice_metric.update(output, target)

    return iou_metric.compute(), dice_metric.compute()

def train_segmentation(model, train_loader, val_loader, optimizer, loss_criterion, epochs=10, pbar=None):
    
    for epoch in range(epochs):
        mean_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = torch.sigmoid(output)
            loss = loss_criterion(output, target.float()) 
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

        iou, dice = validate_segmentation(model, val_loader)
        if pbar is not None:
            pbar.set_description(f'Epoch: {epoch + 1} - Val IoU: {iou:.3f}, Val Dice: {dice:.3f}, train Loss: {mean_loss / len(train_loader)}')
            pbar.update(1)
        else:
            print(f'Epoch: {epoch + 1} - IoU: {iou:.3f}, Dice: {dice:.3f}, Train Loss: {mean_loss / len(train_loader)}')
    # save the model
    torch.save(model, 'segmentation.pth')
    
    

# %%
import segmentation_models_pytorch as smp

segmentation_model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
segmentation_model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, segmentation_model.parameters()), lr=0.0001)
loss_criterion = nn.BCEWithLogitsLoss()
epochs = 100
pbar = tqdm(total=epochs, desc='Training', leave=True, unit='epoch')


train_segmentation(segmentation_model, train_segment_dl, val_segment_dl, optimizer, loss_criterion, epochs=100)

# %%
test_segment_ds[0]

# %%
for image in range(20):
    img, mask = test_segment_ds[image]
    img = img.unsqueeze(0).to(device)
    img_path = test_segment_ds.img_mask_paths[image][0]
    typ = 'malignant' if 'malignant' in img_path else 'benign'
    mask = mask.unsqueeze(0).to(device)
    output = segmentation_model(img)
    output = torch.sigmoid(output)
    output = torch.round(output)
    output = output.int()
    mask = mask.int()

    img = img.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    output = output.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Image {typ}')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground truth')
    plt.subplot(1, 3, 3)
    plt.imshow(output, cmap='gray')
    plt.title('Predicted')
    plt.show()


# %%
for i in test_segment_ds:
    features = radiomics_features(i[0])

# %% [markdown]
# # Random Forests and SVM:

# %%
train_data = np.array([train_classifier_ds.__getitem__(idx)[0].numpy() for idx in range(len(train_classifier_ds))])
test_data  = np.array([test_classifier_ds.__getitem__(idx)[0].numpy() for idx in range(len(test_classifier_ds))])
val_data   = np.array([val_classifier_ds.__getitem__(idx)[0].numpy() for idx in range(len(val_classifier_ds))])

train_labels = np.array([train_classifier_ds.__getitem__(idx)[1] for idx in range(len(train_classifier_ds))]).ravel()
test_labels  = np.array([test_classifier_ds.__getitem__(idx)[1] for idx in range(len(test_classifier_ds))]).ravel()
val_labels   = np.array([val_classifier_ds.__getitem__(idx)[1] for idx in range(len(val_classifier_ds))]).ravel()

# %%
from utils import grid_search

parameters = {'kernel':['linear', 'rbf', 'sigmoid', 'poly', 'random_forest'],
              'C':[0.1, 20],
              'degree':[1, 5],
              'gamma':(0.1, 1, 0.1),
              'criterion': ('gini', 'entropy', 'log_loss'),
              'n_estimators' : (100, 1000, 100),
              'step': 0.3}

best_model, models = grid_search(train_features = train_data,
                    test_features = val_data,   
                    train_labels = train_labels,
                    test_labels = val_labels,
                    params = parameters,
                    folds = 5)

# %%
from sklearn.metrics import accuracy_score

best_model.predict(test_data)
print(f'best model is {best_model.__class__.__name__} with accuracy {accuracy_score(test_labels, best_model.predict(test_data))}')
