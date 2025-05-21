# TamilNet - Enhanced Version

This repository builds upon the original [TamilNet project by Ganesh Manoharan](https://github.com/ganeshmm/TamilNet), a deep learning-based Tamil character recognition system.
My work extends the original model as part of a school project focused on improving performance through architectural and data augmentation enhancements.

I have achived an **test accuracy of 91.4 %** and a **validation accuracy of 97%**.

## 🔍 Project Description

TamilNet is a CNN-based handwritten Tamil character recognition system. In this forked and enhanced version, we aim to **optimize the model's performance** by:

* **Increasing the depth of the neural network** (adding more layers).
* **Applying data augmentation techniques**, specifically rotating input images between **-15° and +15°** in 5 fixed steps to improve generalization.

These enhancements help the model better learn rotational variations in handwriting and improve accuracy on validation/test datasets.

## 🎯 Objective

> Improve character recognition accuracy by training a more expressive model and exposing it to diverse input patterns through geometric data augmentation.

## 🛠️ Key Enhancements

* 🔁 **Data Augmentation**:

  * Rotation from **-15° to +15°** in steps of 5°.
  * Implemented using `torchvision.transforms` pipeline for easy reproducibility.
* 🧠 **Model Architecture Optimization**:

  * Added extra convolutional and fully connected layers.
  * Experimented with different activation functions and batch normalization.

## 📁 Dataset

The original dataset used is the Tamil handwritten character dataset, as specified in the original TamilNet project.

## 🔗 Credits

* Original Work: [TamilNet by Ganesh Manoharan](https://github.com/ganeshmm/TamilNet)
* This enhanced version is part of a **computer vision school project** at HSLU aimed at exploring **images preprocessing and convolution techniques**.