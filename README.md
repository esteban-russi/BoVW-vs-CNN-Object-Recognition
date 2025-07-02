BoVW vs. CNN for Object Recognition
This repository contains a comparative analysis of a traditional Bag of Visual Words (BoVW) model and a deep learning-based Convolutional Neural Network (CNN) for object recognition tasks. The project evaluates their performance, robustness, and suitability for different visual challenges using subsets of the Caltech256 and Human Portraits datasets.
The code is implemented in Python using Scikit-learn for the BoVW pipeline and PyTorch for the CNN.
![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
Table of Contents
Project Overview
Methodologies
1. Bag of Visual Words (BoVW)
2. Convolutional Neural Network (CNN)
Datasets
Performance Results
Installation
Usage
File Structure
Conclusion
Project Overview
Object recognition is a fundamental task in computer vision. This project implements and compares two distinct approaches:
Traditional Approach: A Bag of Visual Words (BoVW) model using SIFT features to create a visual vocabulary and an SVM for classification.
Deep Learning Approach: A Convolutional Neural Network (CNN) that learns hierarchical features directly from pixel data.
The models are first trained and evaluated on a challenging 6-class subset of the Caltech256 dataset. To test for generalization, the optimized models are then evaluated on the Human Portraits dataset, which contains stylized images.
Methodologies
1. Bag of Visual Words (BoVW)
The BoVW pipeline is a traditional method that represents images as histograms of "visual words."
Feature Detection: Scale-Invariant Feature Transform (SIFT) is used to detect keypoints and extract robust local descriptors.
Vocabulary Construction: K-Means clustering is applied to all SIFT descriptors from the training set to create a visual vocabulary. The optimal vocabulary size was found to be 200 visual words.
Image Representation: Each image is converted into a 200-dimensional histogram representing the frequency of each visual word.
Classification: A Support Vector Machine (SVM) with a linear kernel is trained on these histograms to classify the images.
2. Convolutional Neural Network (CNN)
The CNN is a deep learning model designed to automatically learn spatial hierarchies of features.
Architecture: The final, optimized model consists of:
Two convolutional layers (16 and 32 filters, respectively).
A kernel size of 5x5.
ReLU activation functions.
Max-pooling layers for downsampling.
A dropout rate of 0.25 for regularization.
Training: The network was trained using the Adam optimizer with a learning rate of 0.001.
Data Preprocessing: All images for both models were converted to grayscale, resized to 128x128 pixels, and normalized. Data augmentation (rotation) was used to balance the classes in the training set.
Datasets
Caltech256 Subset:
A balanced dataset of 6,000 images was created from six diverse object categories.
Classes: airplanes-101, motorbikes-101, faces-easy-101, t-shirt, billiards, horse.
The dataset was split into 80% training (4800 images) and 20% testing (1200 images).
Human Portraits Dataset:
A dataset from Hugging Face used to evaluate model generalization on stylized images.
Classes: Anime Portrait, Cartoon Portrait, Real Portrait, Sketch Portrait.
A subset of 6,000 images was used to maintain consistency.
Performance Results
The CNN model demonstrated significantly superior performance across both datasets, highlighting its ability to learn more discriminative and robust features.
Model	Dataset	Accuracy (%)	Test Error	Precision	Recall	F1-Score
BoVW	Caltech256	32.0%	0.680	0.320	0.320	0.318
CNN	Caltech256	88.0%	0.120	0.888	0.880	0.882
BoVW	Portraits	64.6%	0.354	0.644	0.647	0.645
CNN	Portraits	99.83%	0.002	0.998	0.998	0.998
<br>
Key Finding: The CNN achieved a 56 percentage-point accuracy improvement over the BoVW model on the complex Caltech256 dataset and a 35 percentage-point improvement on the Portraits dataset.
Installation
Clone the repository:
Generated bash
git clone https://github.com/your-username/bovw-vs-cnn.git
cd bovw-vs-cnn
Use code with caution.
Bash
Create a virtual environment (recommended):
Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Use code with caution.
Bash
Install the required dependencies from requirements.txt:
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
The requirements.txt file should contain:
Generated code
numpy
pandas
matplotlib
scikit-learn
scikit-image
torch
torchvision
datasets
opencv-python-headless
opencv-contrib-python
Pillow
Use code with caution.
Usage
The experiments are structured in Jupyter notebooks or Python scripts, corresponding to the appendices in the original report.
BoVW on Caltech256 (bovw_caltech256.ipynb):
Loads and preprocesses the Caltech256 subset.
Implements SIFT feature extraction.
Builds the visual vocabulary using K-Means.
Trains and evaluates Naive Bayes, kNN, and SVM classifiers.
Visualizes the results, including the final confusion matrix for the SVM model.
CNN on Caltech256 (cnn_caltech256.ipynb):
Loads and preprocesses the same Caltech256 subset.
Defines the CNN architecture in PyTorch.
Includes a hyperparameter search pipeline to find the optimal model configuration.
Trains the final model on the full training set and evaluates its performance on the test set.
Plots learning curves (loss and accuracy).
Comparative Analysis on Portraits Dataset (portraits_evaluation.ipynb):
Loads the Human Portraits dataset.
Applies the best-performing BoVW and CNN models (trained on Caltech256) to this new dataset to test generalization.
Reports and visualizes the final performance metrics for both models on this task.
File Structure
Generated code
.
├── notebooks/
│   ├── bovw_caltech256.ipynb
│   ├── cnn_caltech256.ipynb
│   └── portraits_evaluation.ipynb
├── saved_models/
│   └── best_cnn_model.pth
├── README.md
└── requirements.txt
Use code with caution.
Conclusion
CNN Superiority: Deep learning models like CNNs are vastly superior for complex object recognition tasks. Their ability to learn hierarchical and context-aware features directly from data leads to state-of-the-art performance.
BoVW Limitations: While the BoVW model is conceptually simpler and less computationally demanding, its reliance on handcrafted features (like SIFT) and its disregard for spatial information limit its effectiveness on datasets with high intra-class variability and complex scenes.
Generalization: The CNN also demonstrates exceptional generalization capabilities, achieving near-perfect accuracy on the stylized Portraits dataset, further underscoring the power of learned representations.
