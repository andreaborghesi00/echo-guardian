# Breast Cancer Segmentation – EchoGuardian

This is the final group project developed along with @MirkoMorello and @andypalmi for the AI in Healthcare course at Unimib.
[Canva presentation](https://www.canva.com/design/DAGGIBd_wR4/ZEYYc9pe1K8YqfiOg0X0kw/edit)

## Summary

### Objective

Breast cancer is the most common cancer in women globally, with early detection playing a crucial role in patient survival. This project aims to assist radiologists by developing an AI-powered tool that can automatically segment breast lesions from ultrasound images and classify them as benign or malignant.

---

### Approach

The project is composed of two main tasks:

* **Lesion Segmentation**:
  Semantic segmentation was tackled using state-of-the-art deep learning architectures:

  * **DeepLabV3+** (with ResNet34, ResNet50, and Xception65 backbones)
  * **UNet++** (with ResNet34 backbone)

  These models were trained on 647 ultrasound images paired with pixel-wise lesion masks. Images were augmented during training to improve generalization. For patients with multiple lesions, segmentation masks were merged via bitwise OR.

* **Lesion Classification**:
  Each segmented lesion was analyzed through **radiomic features** (101 per lesion), extracted and used to train three different classifiers:

  * **Support Vector Machine**
  * **Random Forest**
  * **Feed-Forward Neural Network**

  The classification focused on correctly identifying malignant lesions (high sensitivity), with evaluation based on metrics such as F1, sensitivity, specificity, and accuracy.

---

See the slides for visualizations, architecture diagrams, and more technical details.