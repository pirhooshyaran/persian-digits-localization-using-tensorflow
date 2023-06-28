# Persian Digits Localization using TensorFlow

## Introduction
This project focuses on the task of localizing Persian digits. The goal is to accurately detect and localize digits within an image. The project specifically focuses on the localization of digits 3, 5, and 7 for the sake of simplicity. By leveraging machine learning techniques, the project aims to develop a model capable of accurately identifying and bounding the target digits within images.

## Dataset and Data Preparation
The dataset used for this project was prepared for another task and is available in this [GitHub repository](https://github.com/pirhooshyaran/persian-digits-classification-using-LeNet5).
To enhance the dataset and improve the model's performance, each original image was used to create five augmented images. The augmentation process involved randomly selecting the center of each of the five images. The original images were of size (50, 50), while the augmented images were resized to (100, 100). This augmentation step aimed to increase the variability in the dataset and improve the model's ability to generalize. You can see the details in `create_images.py` file.

<p align="left">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/new_images/0001.png" width="100" alt="Image 1">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/new_images/0002.png" width="100" alt="Image 2">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/new_images/0003.png" width="100" alt="Image 3">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/new_images/0004.png" width="100" alt="Image 4">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/new_images/0005.png" width="100" alt="Image 5">
</p>

After preparing the images, the dataset was labeled to provide ground truth information for the localization task. The annotation process was performed using [LabelImg](https://github.com/heartexlabs/labelImg). Annotation files were created to associate bounding box coordinates with the corresponding digits in the images.

## Methodology and Model Architecture
This project utilizes a Convolutional Neural Network (CNN) for the task of localizing Persian digits. The model architecture is implemented using the TensorFlow library.

The CNN model consists of two branches:

Class Prediction Branch: This branch predicts the class of the digit.
Bounding Box Prediction Branch: This branch predicts the bounding box coordinates of the digit.
The architecture follows a common pattern for CNNs, consisting of convolutional layers, max pooling layers, and dense layers. The input images of size (100, 100, 3) are processed through the layers to extract meaningful features.

The class prediction branch uses convolutional layers with 5x5 kernels and max pooling layers with 2x2 pooling. It includes several convolutional layers followed by max pooling and dense layers. The output layer of this branch has 3 units, representing the predicted class probabilities for the digits 3, 5, and 7.

The bounding box prediction branch also uses convolutional layers with 3x3 kernels and max pooling layers with 2x2 pooling. It follows a similar structure with convolutional layers, max pooling layers, and dense layers. The output layer of this branch has 4 units, representing the predicted bounding box coordinates.

These two branches are connected to a common input layer, which takes the input images. The model is trained to simultaneously optimize both the class prediction and bounding box prediction tasks.

<p align="center">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/results/model.png" width="100" alt="Number 2">
</p>

## implementation
In this project, the implementation steps involved:

- Splitting the annotation files: The annotation files were split into separate arrays for class labels and bounding boxes. This was done to create the ground truth for the class output and bounding box output of the model.

- Building the CNN model: The TensorFlow Keras functional API was used to create the CNN model. The model architecture consisted of two branches which is discussed above.

- Defining the IoU metric: To assess the performance of the model's localization task, an Intersection over Union (IoU) metric was defined. This metric measures the overlap between predicted bounding boxes and ground truth bounding boxes.

- Compiling the model: The model was compiled using the Adam optimizer with a learning rate of 0.005. The loss function used for the image classification task was sparse categorical cross-entropy, and the mean squared error was used for the regression task (bounding box prediction). The accuracy metric was used for the image classification task, while the IoU metric was used for the localization task.

- Learning rate scheduler: A learning rate scheduler was implemented to aid in better model convergence. This scheduler adjusted the learning rate during training to help the model find an optimal solution.

- Training the model: The model was initially trained for 5 epochs to observe the initial improvements. Afterwards, the model was trained for an additional 50 epochs to further refine its performance and achieve the final results.

## results
The model was trained for a total of 55 epochs, with an initial training phase of 5 epochs and a subsequent training phase of 50 epochs. The following are the results obtained from the training:

- After 5 epochs: The model achieved an accuracy of 69.412% and an IoU (Intersection over Union) of 61.362%.

<p align="center">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/results/5_epochs.png" width="100" alt="Results after 5 epochs">
</p>

After 50 epochs: The model demonstrated significant improvement, achieving an accuracy of 98.235% and an IoU of 77.791%.

<p align="center">
  <img src="https://github.com/pirhooshyaran/persian-digits-localization-using-tensorflow/blob/master/results/50_epochs.png" width="100" alt="Results after 50 epochs">
</p>

These results indicate that the model has successfully learned to classify and localize the Persian digits. Despite the small size of the dataset used in this project, the achieved performance is quite promising. It is important to note that having a larger dataset would likely lead to even better performance and help mitigate overfitting issues.

The success of the model in this project highlights the potential of leveraging more extensive datasets for commercial-scale tasks. By providing the model with more diverse and abundant data, we can further enhance its performance and generalize it to handle a wider range of real-world scenarios.
