# Behavior-Cloning
cloning behavior of a simulated car using convolution neural network
#### Traffic Sign Recognition

## Writeup

## Author: Vatsal chaturvedi

## Build a Traffic Sign Recognition Project

The goals/steps of this project are the following:

## Load the dataset 

Explore, summarize and visualize the data set
Design, train and test a model architecture
Use the model to make predictions on new images
Analyze the softmax probabilities of the new images
Summarize the results with a written report
# Rubric Points
Here I consider the rubric points individually and describe how I addressed each point in my implementation.

The project code can be found in Traffic_Sign_Classifier.ipynb. The execution results can be found in Traffic_Sign_Classifier.html.

## Dataset Exploration
# Dataset Summary
The basic statistics such as images shapes, number of traffic sign categories, number of samples in training, validation and test image sets are presented in the Step 1: Dataset Summary & Exploration section, A Basic Summary of the Dataset subsection.

# Exploratory Visualization
In section Step 1: Dataset Summary & Exploration, An Exploratory Visualization of the Dataset subsection you can find example images from each category.


# Design and Test a Model Architecture
Preprocessing
Image preprocessing can be found in Step 2: Design and Test a Model Architecture section, Pre-process the Data Set (normalization, grayscale, and so forth) subsection.

There are some transformations required to be performed on each image to feed it to the neural network.

Normalize RGB image. It is done to make each image "look similar" to each other, to make input consistent.
Convert RGB image to grayscale. It was observed that neural network performs slightly better on the grayscale images.


# Model Architecture
The model architecture is defined in Step 2: Design and Test a Model Architecture, Model Architecture subsection. The architecture has 5 layers - 2 convolutional and 3 fully connected. It is LeNet-5 architecture with only one modification - dropouts were added between the layer #2 and layer #3, the last convolutional layer and the first fully connected layer. It was done to prevent neural network from overfitting and significantly improved its performance as a result.

Below is the description of model architecture.

## Layer	Description
Input	32x32x1 gray scale image
Convolution 5x5	1x1 stride, same padding, outputs 28x28x6
RELU	
Max pooling	2x2 stride, outputs 14x14x6
Convolution 5x5	1x1 stride, same padding, outputs 10x10x16
RELU	
Max pooling	2x2 stride, outputs 5x5x16
Flatten	output 400
Drop out	
Fully connected	output 200
RELU	
Fully connected	output 86
RELU	
Fully connected	output 43
# Model Training
The model is using Adam optimizer to minimize loss function.

### learning rate; with 0.0008, 0.0009 and 0.0007 the performance is worse 
RATE       = 0.0010

### number of training epochs; here the model stops improving; we do not want it to overfit
EPOCHS     = 10

### size of the batch of images per one train operation; surprisingly with larger batch sizes neural network reached lower performance
BATCH_SIZE = 128

### the probability to drop out the specific weight during training (between layer #2 and layer #3)
KEEP_PROB  = 0.7

### standart deviation for tf.truncated_normal for weights initialization
STDDEV     = 0.01
Solution Approach

first i tried with learning rate = 0.0007, then i improved it to 0.0008 and with increase in learning rate model's accuracy was increasing
finally i took learning rate of 0.001 and After end of training 
training accuracy was: 0.994 and validation accuracy was: 0.940

The code can be found in section Step 2: Design and Test a Model Architecture, subsection Train, Validate and Test the Model.

# #Test a Model on New Images
# Acquiring New Images

Here are five German traffic signs that were found on the web:
![speed limit 30km/h](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/traffic-sign/traffic-signs/1-Speed-limit-30-km-h.jpg)
![right of the way at next intersection](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/traffic-sign/traffic-signs/11-Right-of-way-at-the-next-intersection.jpg)
![stop](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/traffic-sign/traffic-signs/14-Stop.jpg)
![no vehicles](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/traffic-sign/traffic-signs/15-No-vehicles.jpg)
![vehicles over 3.5t prohibited](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/traffic-sign/traffic-signs/16-Vehicles%20over%203.5t%20prohibited.png)
these images were used for testing model on new traffic signs.
First image "speed limit 30 km/h" might be difficult to identify as there are many similar signs with just change in speed limit.




# Performance on New Images

The code can be found in the section Step 3: Test a Model on New Images, Load and Output the Images subsection.

# Here are the results of the prediction:

# |           PREDICTED                            |       	ACTUAL                                 |
  |  1  Speed limit (30km/h)                       |   1 Speed limit (30km/h)                      |
  |  12 Priority road                              |  16 Vehicles over 3.5 metric tons prohibited  |
  |  1 Speed limit (30km/h)                   	   |  11 Right-of-way at the next intersection     |
  |  14 Stop	                                     |  14 Stop                                      |
  |  15 No vehicles                                |  15 No vehicles                               |
  
  model correctly predicted 3 images out of 5 which gives the test accuracy of 0.6
  The model was not able to correctly classify the the "vehicles over 3.5 metric tons prohibited" sign and "right of the way at next intersection" sign.
  its accuracy is less than what we expected by looking at validation accuracy.

# Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the Step 3: Test a Model on New Images, Top 5 Softmax Probabilities For Each Image Found on the Web subsection.
Probability 	Prediction
0.756710649 	No vehicles
0.122047313 	Speed limit (30km/h)
0.0512931943	Priority road
0.0313977301	Stop
0.0146821784	No passing

[[  9.98717189e-01,   1.04576012e-03,   1.88922262e-04,
          2.08945548e-05,   1.72469299e-05],
       [  5.90810537e-01,   2.99013555e-01,   6.58851713e-02,
          2.71297954e-02,   8.41641519e-03],
       [  9.76065218e-01,   2.19334830e-02,   1.99280749e-03,
          8.25525058e-06,   9.35386240e-08],
       [  2.93564707e-01,   2.67432392e-01,   1.05671450e-01,
          9.22928751e-02,   8.88040513e-02],
       [  9.99999881e-01,   7.67995729e-08,   3.01859977e-08,
          6.17304211e-12,   1.03397098e-12]],
     ([[14, 38, 34, 13, 22],--model is certain that label is 14
       [ 1, 31,  2,  0, 11],--model predicted correctly but softmax probabilities were uncertain
       [12, 16, 40, 41, 42],--model's prediction was incorrect
       [ 1,  2, 12,  5, 15],--model's prediction was incorrect
       [11, 30, 21, 31, 23]]--model's prediction was correct and it was certain that output is 11
