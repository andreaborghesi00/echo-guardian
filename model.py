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
from albumentations.pytorch import ToTensorV2
from SimpleNet import SimpleNet
import segmentation_models_pytorch as smp
import albumentations as A
import torchmetrics
from torchsummary import summary
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torchvision import transforms
import cv2

# %% [markdown]
# ----
# # <center>Data loading

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
print(df.head())

# %% [markdown]
# ## Splitting, datasets, data loaders

# %%
img_mask_paths = [(df['image'].iloc[idx], df['mask'].iloc[idx]) for idx in range(len(df))]
labels = df['label'].values
print(np.shape(labels))

# %%
data_train, data_valtest, label_train, label_valtest = train_test_split(img_mask_paths, labels, test_size=0.2, random_state=69420, stratify=labels)
data_val, data_test, label_val, label_test = train_test_split(data_valtest, label_valtest, test_size=0.5, random_state=69420, stratify=label_valtest)

# %%
# augmentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    # A.Transpose(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    # A.RandomGamma(p=0.5),
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
])

val_transform = A.Compose([
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    A.GaussNoise(p=0.5), # why?
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
# ----
# # <center>Classifier

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice
from torchmetrics import Accuracy, ConfusionMatrix

def validate(model, data_loader):
    
    try:
        pbar.close()
    except:
        pass
    model.eval()
    pbar = tqdm(total=len(data_loader), desc=f"Validation", leave=True)    
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
            pbar.set_description(f"Validation - Runninf Macro Acc: {macro_acc_metric.compute().item():.4f} - Running Micro Acc: {micro_acc_metric.compute().item():.4f}")
            pbar.update(1)
            
    tn, fp, fn, tp = cm.compute().reshape(-1)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    pbar.set_description(f"Validation - Macro Acc: {macro_acc_metric.compute().item():.4f} - Micro Acc: {micro_acc_metric.compute().item():.4f} - Specificity: {specificity:.4f} - Sensitivity: {sensitivity:.4f}")
    pbar.close()
    
    return macro_acc_metric.compute(), micro_acc_metric.compute(), specificity.item(), sensitivity.item()

# %%
from functools import wraps

def train(model, train_loader, val_loader, optimizer, loss_criterion, epochs=10, continue_training=''):
        
    # ----------------------------------- SETUP -----------------------------------
      
    try:
        pbar.close()
    except:
        pass
    previous_epoch = 0
    
    
    try:
        checkpoint = torch.load('./models/best_model.pth')
        best_model = checkpoint['model']
        best_criterion = checkpoint['loss']
        best_scheduler = checkpoint['scheduler']
        best_optimizer = checkpoint['optimizer']
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_criterion.load_state_dict(checkpoint['criterion_state_dict'])
        best_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Found best model, calculating metrics...")
        best_macro, best_micro, best_spec, best_sens = validate(net=best_model, dataloader=val_loader)
        print(f'Best model: Macro Acc: {best_macro:.3f}, Micro Acc: {best_micro:.3f}, Specificity {best_spec:.3f}, Sensitivity {best_sens:.3f}')
        del best_model, best_optimizer, best_criterion, best_scheduler, checkpoint
    except Exception as e:
        best_micro = -1
        print(e)
        print("No best model found, starting from scratch")

    if continue_training != '':
        try:
            checkpoint = torch.load(f'./models/net_{continue_training}.pth')
            model = checkpoint['model']
            loss_criterion = checkpoint['loss']
            optimizer = checkpoint['optimizer']
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_criterion.load_state_dict(checkpoint['criterion_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            previous_epoch = checkpoint['epoch']  # Update previous_epoch
            epochs += previous_epoch  # Update total number of epochs to train
            
            del checkpoint
            print(f"Continuing training of {continue_training} model, checkpoint at epoch {previous_epoch}")
        except Exception as e:
            print(e)
            print(f"No {continue_training} checkpoint found, starting from scratch")

    # ----------------------------------- TRAINING -----------------------------------
    
    for epoch in range(previous_epoch, epochs):
        print ('\n')
        pbar = tqdm(total=len(train_loader), desc=f"Validation", leave=True)
        running_loss = []
        model.train()  
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            pbar.set_description(f"Training - Epoch {epoch}/{epochs}, Running Loss: {np.mean(running_loss):.4f}")
            pbar.update(1)
        
        pbar.set_description(f"Training - Epoch {epoch}/{epochs}, Loss: {np.mean(running_loss):.4f}")
        pbar.close()
        
        model.eval()
        _, micro_acc, _, _ = validate(model, val_loader)
        
        if micro_acc > best_micro:
            print(f"Saving model with Micro Acc: {micro_acc:.3f} > {best_micro:.3f}")
            best_micro = micro_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_state_dict': loss_criterion.state_dict(),
                'model': model,
                'loss': loss_criterion,
                'optimizer': optimizer,
            }, f'./models/best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_state_dict': loss_criterion.state_dict(),
            'model': model,
            'loss': loss_criterion,
            'optimizer': optimizer,
        }, f'./models/net_{continue_training}.pth')



# %%
simple_net = SimpleNet().to(device)
optimizer = optim.Adam(simple_net.parameters(), lr=1e-3, weight_decay=1e-5)
loss_criterion = nn.BCELoss()
to_train = False

if to_train:
    train(model = simple_net, train_loader = train_classifier_dl, val_loader = val_classifier_dl, optimizer = optimizer, loss_criterion = loss_criterion, epochs = 100, continue_training='simple_net')
else:
    simple_net.load_state_dict(torch.load('./models/net_simple_net.pth')['model_state_dict'])

# %% [markdown]
# ----
# # <center>Segmentation

# %% [markdown]
# ## Data preparation

# %%
import cv2

# Create new dataset folder with unique masks for each image
new_path = 'dataset_unique_masks/'

for folder in os.listdir(new_path):
    for file in os.listdir(os.path.join(new_path, folder)):
        if 'mask_' in file:
            original_mask = file.split('_mask_')[0] + '_mask.png'
            print(f'original_mask = {original_mask}')
            print(f'second mask = {file}')
            orig_mask = cv2.imread(os.path.join(new_path, folder, original_mask), cv2.IMREAD_GRAYSCALE)
            second_mask = cv2.imread(os.path.join(new_path, folder, file), cv2.IMREAD_GRAYSCALE)
            new_mask = cv2.bitwise_or(orig_mask, second_mask)
            cv2.imwrite(os.path.join(new_path, folder, original_mask), new_mask)
            # !rm "{os.path.join(new_path, folder, file)}"

path = 'dataset_unique_masks/'

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
print(df_segment.shape)

# %%
img_mask_paths_segment = [(df_segment['image'].iloc[idx], df_segment['mask'].iloc[idx]) for idx in range(len(df_segment))]
labels = df_segment['label'].values
print(np.shape(labels))
# print(labels)

data_train, data_valtest, label_train, label_valtest = train_test_split(img_mask_paths_segment, labels, test_size=0.2, random_state=69420, stratify=labels)
data_val, data_test, label_val, label_test = train_test_split(data_valtest, label_valtest, test_size=0.5, random_state=69420, stratify=label_valtest)

# %%

segmentation_train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    #A.RandomGamma(p=0.5),
    #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=50.0, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
    A.Resize(256, 256),
    ToTensorV2(),
])

# pytorch transforms
segmentation_valtest_transform = A.Compose([
    ToTensorV2(),
])

# %%
# Create and save or load the dataloaders
batch_size = 24
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
                                    exclusion_class='cnn',
                                    scaler = None)
    
    val_segment_ds = SegmentationDataset(data_val, label_val,
                                    transform=segmentation_valtest_transform,
                                    json_exclude_path='excludedImages.json',
                                    exclusion_class='cnn',
                                    scaler = None)
    
    test_segment_ds = SegmentationDataset(data_test, label_test,
                                    transform=segmentation_valtest_transform,
                                    json_exclude_path='excludedImages.json',
                                    exclusion_class='cnn',
                                    scaler = None)
    torch.save(train_segment_ds, train_segment_ds_path)
    torch.save(test_segment_ds, test_segment_ds_path)
    torch.save(val_segment_ds, val_segment_ds_path)
    print('Saved train_segment_ds, val_segment_ds and test_segment_ds to file')
    
train_segment_dl = DataLoader(train_segment_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_segment_dl = DataLoader(test_segment_ds, batch_size=batch_size, num_workers=8)
val_segment_dl = DataLoader(val_segment_ds, batch_size=batch_size, num_workers=8)


# %% [markdown]
# ## Train definition

# %%
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss - {config}")
    plt.legend()
    plt.show()


# %%
import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

def apply_crf(image, output, iterations=5):
    # Assuming the image is in the shape (C, H, W) and output is in the shape (n_classes, H, W)
    c, h, w = image.shape[-3:]
    n_classes = output.shape[0]
    
    # Reshape the output to (n_classes, H * W)
    output = output.reshape((n_classes, -1))
    
    d = dcrf.DenseCRF2D(w, h, n_classes)
    unary = unary_from_softmax(output)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    # Create pairwise bilateral energy
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=image.transpose((1, 2, 0)), chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    
    q = d.inference(iterations)
    res = np.argmax(q, axis=0).reshape((h, w))
    
    return res


# %%
from functools import wraps
from tqdm import tqdm
import torch
from torchmetrics.classification import BinaryJaccardIndex, Dice


def validate_segmentation(model, data_loader):
    iou_metric_val = BinaryJaccardIndex().to(device)
    dice_metric_val = Dice(num_classes=1, multiclass=False).to(device)
    val_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_criterion(output, target.float())
            val_loss += loss.item()
            output = torch.sigmoid(output)
            output = torch.round(output)
            output = output.int()
            target = target.int()
            iou_metric_val.update(output, target)
            dice_metric_val.update(output, target)
    return iou_metric_val.compute(), dice_metric_val.compute(), val_loss


def train_segmentation(model, train_loader, val_loader, optimizer, loss_criterion, scheduler, epochs=10, continue_training=''):
    # ----------------------------------- SETUP -----------------------------------
    try:
        pbar.close()
    except:
        pass
    previous_epoch = 0

    try:
        checkpoint = torch.load('./models/best_segmentation_model.pth')
        best_model = checkpoint['model']
        best_criterion = checkpoint['loss']
        best_optimizer = checkpoint['optimizer']
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_criterion.load_state_dict(checkpoint['loss_state_dict'])
        best_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Found best model, calculating metrics...")
        best_iou, best_dice = validate_segmentation(best_model, val_loader)
        print(f'Best model: IoU: {best_iou:.3f}, Dice: {best_dice:.3f}')
        del best_model, best_optimizer, best_criterion, checkpoint
    except Exception as e:
        best_iou = -1
        print(e)
        print("No best model found, starting from scratch")

    if continue_training != '':
        try:
            checkpoint = torch.load(f'./models/segmentation_model_{continue_training}.pth')
            model = checkpoint['model']
            loss_criterion = checkpoint['loss']
            optimizer = checkpoint['optimizer']
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_criterion.load_state_dict(checkpoint['criterion_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            previous_epoch = checkpoint['epoch']
            epochs += previous_epoch

            del checkpoint
            print(f"Continuing training of {continue_training} model, checkpoint at epoch {previous_epoch}")
        except Exception as e:
            print(e)
            print(f"No {continue_training} checkpoint found, starting from scratch")

    # ----------------------------------- TRAINING -----------------------------------
    iou_metric_train = BinaryJaccardIndex().to(device)
    dice_metric_train = Dice(num_classes=1, multiclass=False).to(device)

    train_losses = []
    val_losses = []

    for epoch in range(previous_epoch, epochs):
        print('\n')
        pbar = tqdm(total=epochs, desc=f"Training - Epoch {epoch}/{epochs}", leave=True, unit='epoch')
        train_loss = 0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            output = torch.sigmoid(output)
            output = torch.round(output)
            output = output.int()
            target = target.int()
            iou_metric_train.update(output, target)
            dice_metric_train.update(output, target)

        iou_train = iou_metric_train.compute()
        dice_train = dice_metric_train.compute()
        iou_metric_train.reset()
        dice_metric_train.reset()

        scheduler.step()

        model.eval()
        iou_val, dice_val, val_loss = validate_segmentation(model, val_loader)

        # Plot the loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 25 == 0:
            # Plot the loss
            plot_loss(train_losses, val_losses)

        pbar.set_description(f"Training - Epoch {epoch}/{epochs}, Loss: {train_loss:.4f} | Train IoU: {iou_train:.3f} - Val IoU {iou_val:.3f} | Train Dice: {dice_train:.3f} - Val Dice {dice_val:.3f}")
        pbar.update(1)
        pbar.close()

        if dice_val > best_iou:
            print(f"Saving model with Dice Val: {dice_val:.3f} > {dice_val:.3f}")
            best_iou = dice_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_state_dict': loss_criterion.state_dict(),
                'model': model,
                'loss': loss_criterion,
                'optimizer': optimizer,
            }, f'./models/best_segmentation_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_state_dict': loss_criterion.state_dict(),
            'model': model,
            'loss': loss_criterion,
            'optimizer': optimizer,
        }, f'./models/segmentation_model_{continue_training}.pth')


# %%
ALPHA = 0.8
GAMMA = 1

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


# %%
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to the inputs to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute the intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        # Compute the Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Compute the Dice loss
        dice_loss = 1.0 - dice_coeff

        return dice_loss


# %%
class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = FocalLoss()
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets)
        combined_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return combined_loss


# %%
from VisionTransformer import TransformerEncoder, TransformerUNet


# Create the Transformer encoder
encoder = TransformerEncoder(
    img_size=256,
    patch_size=8,
    in_chans=1,
    embed_dim=768,
    depth=12,
    n_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    p=0.1,
    attn_p=0.1,
)

config = {
    'arch': 'DeepLabV3Plus',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 1,
    'classes': 1,
    'img_size': 256,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'n_heads': 16,
    'mlp_ratio': 4,
    'qkv_bias': True,
    'p': 0.,
    'attn_p': 0.,
}
# Create the Transformer U-Net model
#transformer_vision = TransformerUNet(encoder, out_chans=1)
transformer_unet = TransformerUNet(config)

transformer_unet.to(device)

# %%
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from ranger21 import Ranger21
#import optuna

config = {
    'arch': 'DeepLabV3Plus',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 1,
    'classes': 1,
}

#segmentation_model = smp.create_model(**config)
segmentation_model = transformer_unet

segmentation_model.to(device)

learning_rate = 3e-4
weight_decay = learning_rate * 0.05
epochs = 200
to_train = True

# Implement Cosine Annealing Learning Rate
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, segmentation_model.parameters()), lr=learning_rate, weight_decay=weight_decay)

# Update optimizer to Ranger
num_batches_per_epoch = len(train_segment_dl)

#Advanced optimizer and scheduler
# optimizer = Ranger21(
#     transformer_vision.parameters(),
#     lr=learning_rate,
#     weight_decay=weight_decay,
#     num_epochs=epochs,
#     num_batches_per_epoch=num_batches_per_epoch
# )

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs/4), T_mult=2, eta_min=learning_rate/10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/10)
#scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_segment_dl), epochs=epochs)

# Regularization in the transformer blocks and decoder layers
# for block in transformer_unet.encoder.blocks:
#     block.mlp.add_module('dropout', nn.Dropout(p=0.1))

# for layer in transformer_unet.decoder:
#     if isinstance(layer, nn.Sequential):
#         layer.add_module('dropout', nn.Dropout(p=0.1))

loss_criterion = nn.BCEWithLogitsLoss()
#loss_criterion = CombinedLoss(weight_dice=0.3, weight_ce=0.7)
#loss_criterion = FocalLoss()
# use the dice loss
#loss_criterion = DiceLoss()

if to_train:
    train_segmentation(segmentation_model, train_segment_dl, val_segment_dl, optimizer, loss_criterion, scheduler, epochs=epochs, continue_training='')

# %%
iou_test, dice_test, val_loss = validate_segmentation(segmentation_model, test_segment_dl)
print(f'Test IoU: {iou_test:.3f}, Test Dice: {dice_test:.3f}')

# %%
model_name = config['arch'] + '_' + config['encoder_name'] + '_lr_' + str(config['lr']) + '_epochs_' + str(config['epochs']) + '_model.pth'
model_path = os.path.join('models', model_name)
torch.save(segmentation_model.state_dict(), model_path)

df_path = os.path.join('models', 'models.csv')
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    df = pd.DataFrame(columns=['arch', 'encoder_name', 'encoder_weights', 'in_channels', 'classes',
                               'optimizer', 'lr', 'weight_decay', 'epochs', 'test_iou', 'test_dice'])

# Add the current model's configuration to the DataFrame
df = pd.concat([df, pd.DataFrame({
    'arch': [config['arch']],
    'encoder_name': [config['encoder_name']],
    'encoder_weights': [config['encoder_weights']],
    'in_channels': [config['in_channels']],
    'classes': [config['classes']],
    'optimizer': [config['optimizer']],
    'lr': [config['lr']],
    'weight_decay': [config['weight_decay']],
    'epochs': [config['epochs']],
    'test_iou': [iou_test.cpu().numpy()], 
    'test_dice': [dice_test.cpu().numpy()]
})])

# Save the DataFrame
df.to_csv(df_path, index=False)

# %% [markdown]
# ----
# # <center> Test segmentation and classification in tandem

# %%
from NNClassification import NNClassifier
from UnetSegmenter import UnetSegmenter
from PIL import Image

classifier = NNClassifier('models/best_model.pth')
segmenter = UnetSegmenter('models/segmentation_model_.pth') 


# test of the classifier and segmenter by opening an image and mask from file
image = Image.open('dataset/malignant/malignant (1).png').convert('L')
mask = Image.open('dataset/malignant/malignant (1)_mask.png')
prediction = classifier.predict(image = image, mask = mask)
masked_prediction = segmenter.predict(image = image)
plt.imshow(masked_prediction)
image.close()
mask.close()

print(prediction[0][0].cpu().numpy())

# %%
import json
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


i = 0
real_labels = []
predicted_labels = []
predicted_labels_generated_mask = []

for image in range(len(test_segment_ds)):
    with open('excludedImages.json', 'r') as f:
        json_exclude_path = json.load(f)
        if 'benign' in test_segment_ds.img_mask_paths[image][1] and int(re.findall(r'\d+', test_segment_ds.img_mask_paths[image][1])[0]) in json_exclude_path['cnn']['benign']:
            continue
        elif 'malignant' in test_segment_ds.img_mask_paths[image][1] and int(re.findall(r'\d+', test_segment_ds.img_mask_paths[image][1])[0]) in json_exclude_path['cnn']['malignant']:
            continue
    i += 1
    img, mask = test_segment_ds[image]
    img = img.squeeze().cpu().numpy() # remove the tensor dimension
    mask = mask.squeeze().cpu().int().numpy()
    
    predicted_mask = segmenter.predict(image = img * 255)
    try:
        predicted_label_generated_mask = classifier.predict(image = img, mask = predicted_mask)[0][0]
    except:
        continue
    predicted_label_real_mask = classifier.predict(image = img, mask = mask)[0][0]
    
    real_label = 'benign' if 'benign' in test_segment_ds.img_mask_paths[image][0] else 'malignant'
    predicted_label_real_mask_l = 'benign' if np.round(predicted_label_real_mask.cpu()) == 0 else 'malignant'
    predicted_label_generated_mask_l = 'benign' if np.round(predicted_label_generated_mask.cpu()) == 0 else 'malignant'
    
    real_labels.append(1 if real_label == 'malignant' else 0)
    predicted_labels.append(predicted_label_real_mask.cpu())
    predicted_labels_generated_mask.append(predicted_label_generated_mask.cpu())
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Image real\nreal label: {real_label}')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask ground truth\npredicted label: {predicted_label_real_mask_l}\n p = {predicted_label_real_mask :.2f}')
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title(f'Predicted mask\npredicted label: {predicted_label_generated_mask_l}\n p = {predicted_label_generated_mask:.2f}')
    plt.show()


print(f'Accuracy with real mask: {accuracy_score(real_labels, np.round(predicted_labels))}')

cm = confusion_matrix(real_labels, np.round(predicted_labels))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(f'Accuracy with generated mask: {accuracy_score(real_labels, np.round(predicted_labels_generated_mask))}')

cm = confusion_matrix(real_labels, np.round(predicted_labels_generated_mask))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




# %%
from cryptography.fernet import Fernet
from torchvision import io

def encrypt_model(model_path, key):
    
    model = torch.load(model_path)
    model_bytes = torch.save(model, None)
    cipher = Fernet(key)
    encrypted_model_bytes = cipher.encrypt(model_bytes)
    
    with open('encrypted_model.pth', 'wb') as file:
        file.write(encrypted_model_bytes)

key = b'' # has to be 32 url-safe base64-encoded bytes, pad if not

if len(key) < 32:
    key += b'=' * (32 - len(key))
elif len(key) > 32:
    key = key[:32]
encrypt_model('model.pth', key)


# %%
def decrypt_model(encrypted_model_path, key):
    with open(encrypted_model_path, 'rb') as file:
        encrypted_model_bytes = file.read()
    
    cipher = Fernet(key)
    try:
        decrypted_model_bytes = cipher.decrypt(encrypted_model_bytes)
        model = torch.load(io.BytesIO(decrypted_model_bytes))
        return model
    except:
        print("Invalid key or corrupted model file.")
        return None

key = b''

if len(key) < 32:
    key += b'=' * (32 - len(key))
elif len(key) > 32:
    key = key[:32]

model = decrypt_model('encrypted_model.pth', key)

if model is not None:
    model.eval()

# %% [markdown]
# ----
# # <center>Random Forests and SVM:

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
