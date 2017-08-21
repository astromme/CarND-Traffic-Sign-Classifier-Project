# Traffic Sign Recognition

Build a Traffic Sign Classifier ([project code](https://github.com/astromme/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb))

[//]: # (Image References)

[visualization1]: ./examples/visualization.jpg "Visualization"
[grayscaling]: ./examples/grayscale.jpg "Grayscaling"
[random_noise]: ./examples/random_noise.jpg "Random Noise"
[accuracy_loss]: ./examples/accuracy_loss.png "Accuracy & Loss"
[dataset_exploration1]: ./examples/dataset-exploration1.png "Dataset Exploration 1"
[dataset_exploration2]: ./examples/dataset-exploration2.png "Datset Exploration 2"
[augmented_comparison]: ./examples/dataset-augmentation1.png "Augmented Comparison"
[network]: ./examples/network.png "Network"

[image1]: ./test_photos/photo1.png "Test Traffic Sign 1"
[image2]: ./test_photos/photo2.png "Test Traffic Sign 2"
[image3]: ./test_photos/photo3.png "Test Traffic Sign 3"
[image4]: ./test_photos/photo4.png "Test Traffic Sign 4"
[image5]: ./test_photos/photo5.png "Test Traffic Sign 5"
[image6]: ./test_photos/photo6.png "Test Traffic Sign 6"
[image7]: ./test_photos/photo7.png "Test Traffic Sign 7"
[image8]: ./test_photos/photo8.png "Test Traffic Sign 8"
[image9]: ./test_photos/photo9.png "Test Traffic Sign 9"

## Rubric Points
### Data Set Summary & Exploration

Basic stats about the dataset:

* Number of training examples = 34799
* Number of training examples = 4410
* Number of testing examples = 12630
* Image data shape = [32, 32]
* Number of classes = 43

Examples of each class in the dataset:
![alt text][dataset_exploration1]

Distribution of classes in the training (green), validation (red), and test (blue) data sets:
![alt text][dataset_exploration2]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


My most successful model architecture and data processing steps are actually very simple. The data processing converts images to grayscale and normalizes values to the range [-1, 1]. The model has two convolutional layers each followed by max pooling, dropout, and relu activation. Finally there are two fully connected layers, each having relu activation. This produced an accuracy of 93% on the validation dataset.

I tried other data augmentation, but this failed to increase the accuracy past 93%. Specifically, I tried generating 5x additional training data with random jitter including:

* translations (mean: 0.0, sigma: 2.0 pixels)
* rotations (mean: 0.0, sigma: 10.0 degrees)
* scale (mean: 1.0, sigma: 0.1x)
* brightness
* contrast

This seemed to reduce overfitting because now the training set accuracy was below the test set accuracy, but never surpassed 93% accuracy even after thousands of epochs.

Here is an example of an augmented image (top) and an original image (bottom):

![alt text][augmented_comparison]

Even though the simple network performed best, I also tried more complicated networks, including the following:

* 3 convolution layers instead of two
* connecting the outputs of the last two convolutions to the fully connected output layers, rather than just the last convolution
* changing dropout keep_prob between 0.5 and 1.0
* changing the depth of each convolutional layer (e.g. double, half)
* changing the number of fully connected units (e.g. double, half)
* removing the max_pool layers
* changing the kernel size of the convolutional layers to 3,3 instead of 5,5
* changing the padding from VALID to SAME

To help understand the results better, I also added tensorboard integration and logged summaries of the convolutional variables & the validation & training error/loss.


My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16				|
| RELU					|												|
| Dropout					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32			|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs: 800, outputs: 120       									|
| RELU					|												|
| Fully connected		| inputs: 120, outputs: 84       									|
| RELU					|												|
| Fully connected		| inputs: 84, outputs: 43 (num_classes)       									|
| Softmax				|         		predictions							|


![network comparison][network]



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters:

|Hyperparemter|Value|Notes|
|:-----------:|:---:|:---:|
|Batch size   |2048 |This is close to the maximum that fits on my GPU|
|Epochs       |600 | |
|Learning Rate|0.001|Decreasing this didn't seem to increase the accuracy|
|keep_prob|0.6|only for training|

I used the Adam optimizer.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 9 German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5]![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


resample...

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
