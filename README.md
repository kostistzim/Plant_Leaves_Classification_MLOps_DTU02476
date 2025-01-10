# Plant_Leaves_Classification

## Group 96 - Team Members

Kostis Tzimoulias - s242796@dtu.dk  
Xhino Mullaymeri - s223481@dtu.dk  
Dimitris Papantzikos - s242798@dtu.dk  
Jan Šulíček - s246733@dtu.dk  
Michail Dikaiopoulos - s242816@dtu.dk  

## Goal

Our goal for the project is to implement a binary classification model for classifying a dataset of plant leaves as being in healthy or diseased condition. We aim to use a pretrained Convolutional Neural Network implementation and finetune it to achieve the best results. Our main focus is to deliver and deploy a model that is well organized, fully reproducible, easy-to-interact with and aligned with the overall direction of the course. To do that, we are going to use all the tools and techniques that we are currently learning in the course.

## Framework

We will be using PyTorch, with the addition of other third party libraries. PyTorch Image Models (timm) contains pre-trained image classification neural networks. Torchvision will be used to handle image augmentation, both to enhance the size of the dataset and to make the model more robust in the end. Hydra provides us an elegant way to handle our hyperparameters.

## Data

As for the dataset, we are going to use [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) from Kaggle.

The dataset contains 4502 images of healthy and unhealthy plant leaves with an approximate size of 7.3GBs. We will start by scaling down the images from (6000*4000) to the model's proper input dimensions. In case we notice that the size of the dataset requires computational expensive training, we will utilize a subset of the initial data.

## Models

We are going to use CNN based model(s), more specifically [timm/efficientnet_b1.ra4_e3600_r240_in1k - Hugging Face](https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k).

The idea is to use the CNN body of this pretrained model and blend it with our own fully connected classification layer(s).
