# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/class_distribution_chart.png "Traffic sign class distribution"
[image2]: ./examples/preprocessing.png "Preprocessing results"
[image3]: ./examples/augmentation.png "Augmentation results"
[image4]: ./examples/class_distribution_chart_after_augmentation.png "Traffic sign class distribution after augmentation"
[image5]: ./examples/probabilities_summary.png "Probability Summary"
[sample01]: ./custom_images/sample01.png "Traffic Sign 1"
[sample02]: ./custom_images/sample02.png "Traffic Sign 2"
[sample03]: ./custom_images/sample03.png "Traffic Sign 3"
[sample04]: ./custom_images/sample04.png "Traffic Sign 4"
[sample05]: ./custom_images/sample05.png "Traffic Sign 5"
[image6]: ./examples/top5_softmax_sample01.png "The top five softmax probabilities of the prediction for image 1"
[image7]: ./examples/top5_softmax_sample02.png "The top five softmax probabilities of the prediction for image 2"
[image8]: ./examples/top5_softmax_sample03.png "The top five softmax probabilities of the prediction for image 3"
[image9]: ./examples/top5_softmax_sample04.png "The top five softmax probabilities of the prediction for image 4"
[image10]: ./examples/top5_softmax_sample05.png "The top five softmax probabilities of the prediction for image 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

*Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43


* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed depending on the traffic sign type.

![image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing approach that we use is the following one: 
1. Converting images to grayscale
2. Normalize the images
3. Equalized the image histogram


The image shows from left to right how the different transformations affected the dataset:

![image2]

Due to the nature of the images, it's very simple and straightforward to add extra data to improve our model. I applied two different transformations to the data:

1. Projection transform
2. Rotation

The image below shows what are the output of the different transformations (from left to right: original image, rotated image, perspective transformed image):

![image3]

These transformation will be applied randomly to samples of classes represented by less than 1500 samples, and will be added to that class' set until a minimum of 1500 samples is fulfilled.

Here is the same label distribution chart shown below updated with the new populated dataset:

![image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16   |
| RELU                  |           									|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Convolution 1x1       | 2x2 stride, valid padding, outputs 1x1x400	|			
| RELU					|												|
| Fully connected       | Input 400, Output 120                         |
| RELU 	                |                                               |
| Dropout               | 50% (only in training)                        |
| Fully connected       | Input 120, Output 84                          |
| RELU                  | 	                                            |
| Dropout               | 50% (only in training)                        |
| Fully connected       | Input 84, Output 43                           |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train my model, I used the Adam Optimizer and a learning rate of 0.001. I used a batch size of 128 and 300 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.933
* test set accuracy of 0.965

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First architecture was the default LeNet network displayed in the earlier exercise in the lesson.
* What were some problems with the initial architecture?
Accuracy was lower than the required minimum of 94%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Added an extra layer and added dropouts.
* Which parameters were tuned? How were they adjusted and why?
Extended the number of epochs.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Dropout layers helped to avoid overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![sample01] ![sample02] ![sample03] 
![sample04] ![sample05]

There are several difficulties to classify these images, the resolution, the perspective, different contrast and brightness, ... All these difficulties are overcome by applying data preprocessing and augmented data.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Ahead only      		| Ahead only   									| 
| Go straight or right	| Go straight or right				            |
| Priority road	   		| Priority road					 				|
| No entry      		| No entry              						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test, and validation sets.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section **Predict the Sign Type for Each Image and Analyze Performance** of the Jupyter notebook.

Below is a summary of the top 5 softmax probabilities provided for each sign:

![image5]

And the probability distribution for each of the images:

![image6]
![image7]
![image8]
![image9]
![image10]

The fact that no other probabilities are even showing up in the chart shows that the predictions were very exact for those cases.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


