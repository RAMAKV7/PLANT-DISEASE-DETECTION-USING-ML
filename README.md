# PLANT-DISEASE-DETECTION-USING-ML
Plant disease detection

P.s : FOR PROMISING RESULTs ON IMAGE BASED CLASSIFICATION USE DEEP LEARNING INSTEAD OF NORMAL MACHINE LEARNING ALGORITHMS.

Plant Disease Detection is one of the mind boggling issue that exits when we talk about using Technology in Agriculture.Although researches has been done to detect weather a plant is healthy or diseased using Deep Learning and with the help of Neural Network, new techniquies are still being discovered.

Here is my approach for Detecting weather a plant leaf is healthy or unhealthy by utilising classical Machine Learning Algorithm , Pre-processing the data using Image Processing.

 DATASET INFORMATION

The data used for this project is extracted from the folder named “color” which is situated in the folder named “raw” in the Github Repository. The Data fed for the modeling is of Apple Leaves. For training purpose the Dataset comprises of 2 folders named Diseased and Healthy which contains images of leaves with respective labels. The Diseased Folder contains diseased/unhealthy, affected by Apple Scab, Black Rot or Cedar Apple Rust. The Healthy Folder consists of Green and healthy images.

PROPERTIES OF IMAGES
Type of File : JPG File.

Dimensions : 256 * 256.

Width : 256 Pixels.

Height : 256 Pixels.

Horizontal Resolution : 96 dpi.

Vertical Resolution : 96 dpi.

Bit Depth : 24.

STEPS INVOLVED

Data Preprocessing

1) Load Original Image. A total of 800 images for each class Diseased and Healthy is fed for the machine.

2)  Conversion of image from RGB to BGR. Since Open CV (python library for Image Processing), accepts images in RGB coloring format so it needs to be converted to the original format that is BGR format.

3)  Conversion of image from BGR to HSV. The simple answer is that unlike RGB, HSV separates luma, or the image intensity, from chroma or the color information. This is very useful in many applications. For 
    example, if you want to do histogram equalization of a color image, you probably want to do that only on the intensity component, and leave the color components alone. Otherwise you will get very strange 
    colors. In computer vision you often want to separate color components from intensity for various reasons, such as robustness to lighting changes, or removing shadows. Note, however, that HSV is one of many 
    color spaces that separate color from intensity (See YCbCr, Lab, etc.). HSV is often used simply because the code for converting between RGB and HSV is widely available and can also be easily implemented.

4)  Image Segmentation for extraction of Colors. In order to separate the picture of leaf from the background segmentation has to performed, The color of the leaf is extracted from the image.

5)  Applying Global Feature Descriptor. Global features are extracted from the image using three feature descriptors namely :

         • Color : Color Channel Statistics (Mean, Standard Deviation) and Color Histogram

         • Shape : Hu Moments, Zernike Moments

         • Texture : Haralick Texture, Local Binary Patterns (LBP)

           After extracting the feature of images the features are stacked together using numpy function “np.stack”.

           According to the images situated in the folder the labels are encoded in numeric format for better understanding of the machine.

           The Dataset is splitted into training and testing set with the ratio of 80/20 respectively.

6)  Feature Scaling Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes 
    or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the 
    values.Here, we have used Min-Max Scaler. This scaling brings the value between 0 and 1.

7)  Saving the Features. After features are extracted from the images they are saved in HDF5 file. The Hierarchical Data Format version 5 (HDF5), is an open source file format that supports large, complex, 
    heterogeneous data. HDF5 uses a "file directory" like structure that allows you to organize data within the file in many different structured ways, as you might do with files on your computer.

8)  Modeling The Model is trained over 7 machine learning models named :

         • Logistic Regression

         • Linear Discriminant Analysis

         • K Nearest Neighbours

         • Decision Trees

         • Random Forest

         • Naïve Bayes

         • Support Vector Machine

          And the model is validated using 10 k fold cross validation technique.

9 ) Prediction The models with best performance is them trained with whole of the dataset and score for testing set is predicted using Predict function.

    An accuracy of 97% is achieved using Randomm Forest Classifier.

    What and Where Specification
    Utils : Contains python file for conversion of labels of images in the train folders.

    Image Classification : Contains Training Dataset and the .ipynb for the Plant Disease Detection.
 
    Testing Notebook : Contains Detailed Specification of Functions applied in the leaf images.

COMPARISON WITH THE EXISTING SYSTEM

1. PlantVillage Dataset (Penn State University)

Description: PlantVillage, a project by Penn State University, provides a large dataset of images for various plant diseases. This dataset is one of the most commonly used for CNN-based disease classification.
Strengths:
Large-scale dataset with over 50,000 labeled images.
Covers diseases in over 14 crops including tomatoes, potatoes, and apples.
Open-access, widely adopted in academic research and industry applications.
Weaknesses:
Some images have variations in quality and lighting, which may affect model performance.
Limited coverage of certain diseases, particularly in tropical plants and small-scale crops.
Key Features:
Pre-labeled images for both healthy and diseased plants.
Highly used for benchmarking CNN-based plant disease detection models.

2. DeepPlant (2017)

Description: DeepPlant is a deep learning framework designed for the automatic identification of plant diseases through leaf images. It uses a combination of CNNs and deep neural networks for enhanced accuracy.
Strengths:
High accuracy with CNN architectures, using image features like texture, color, and shape for classification.
Uses pre-trained networks like AlexNet and VGGNet, making it faster to train on new datasets.
Robust to environmental changes, which is crucial for real-world applications.
Weaknesses:
Requires a large number of training images for effective learning.
Potential overfitting on smaller datasets, especially with limited image diversity.
Key Features:
Achieved high accuracy in plant disease classification tasks.
Successfully implemented on mobile and low-power devices for real-time applications.

3. Plant Disease Classification with Convolutional Neural Networks (2018)

Description: This system applies CNNs to a variety of plant disease detection tasks, with a focus on optimizing architectures for high performance on datasets like PlantVillage.
Strengths:
High performance achieved using modern CNN architectures such as ResNet and DenseNet.
Provides detailed classification metrics like F1 score, accuracy, and confusion matrix to evaluate the model's robustness.
Weaknesses:
Requires high computational resources and specialized hardware (e.g., GPUs) for training on large datasets.
May be prone to class imbalance in multi-class classification, affecting the precision and recall of certain diseases.
Key Features:
Focus on model optimization and performance benchmarking.
Utilizes advanced techniques like data augmentation to reduce overfitting.

4. PlantDoc: A Plant Disease Diagnosis System (2020)

Description: PlantDoc is a plant disease detection system that uses CNN-based models to classify diseases based on images. It is designed to be integrated into a mobile application.
Strengths:
Specifically designed for mobile platforms, optimizing the model for on-device inference.
High-speed inference suitable for real-time disease detection.
Focuses on ease of use and integration with user-friendly mobile apps.
Weaknesses:
Limited coverage of rare or less common plant diseases.
Dependency on mobile device specifications for real-time inference performance.
Key Features:
Pre-trained models optimized for mobile deployment.
Good accuracy for common plant diseases with a user-centric design for plant health tracking.

5. Plant Disease Detection using Deep Convolutional Neural Networks (2022)

Description: This model leverages the power of CNNs to detect a wide variety of plant diseases, focusing on optimizing performance across different plant types and disease variations.
Strengths:
Utilizes advanced CNN architectures like VGG16, EfficientNet, and ResNet for improved accuracy.
Achieves high classification accuracy on large, diverse datasets.
Incorporates multi-class classification techniques for more detailed results.
Weaknesses:
May require fine-tuning and hyperparameter optimization to adapt to specific crops and diseases.
Large models may be computationally expensive for real-time applications.
Key Features:
Improved classification with multi-layer neural networks.
Strong ability to generalize across different plant species and disease types.

CONCLUSION

Existing systems for plant disease detection using CNNs exhibit various strengths, such as high accuracy, robustness to environmental changes, and scalability. However, they also face challenges like the need for large datasets, high computational power, and limited coverage of rare plant diseases. The choice of system largely depends on the specific requirements, such as real-time detection, mobile deployment, or research-oriented classification tasks. This project aims to address some of these challenges by providing a more efficient, scalable, and user-friendly solution for plant disease detection.

