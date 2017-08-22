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
[featuremaps-conv1]: ./examples/featuremaps-conv1.png "featuremaps conv1"
[featuremaps-conv2]: ./examples/featuremaps-conv2.png "featuremaps conv2"

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

My most successful model architecture and data processing steps combines 3 convolutional layers with data normalization and augmentation. This produces an accuracy of 98% on the validation dataset and 96% on the testing dataset.

#### Data normalization

I convert images to grayscale and normalize to between [0.5, -0.5]. This both speeds up training and makes training more effective.

#### Data augmentation

I generate 5x additional training data (so, 6x data total) with random jitter including:

* translations (mean: 0.0, sigma: 2.0 pixels)
* rotations (mean: 0.0, sigma: 10.0 degrees)
* scale (mean: 1.0, sigma: 0.1x)
* brightness
* contrast

This better represents the range of real life scenarios and significantly reduces overfitting.

Here is an example of an augmented image (top) and an original image (bottom):

![alt text][augmented_comparison]

#### Network architecture

Before settling on a network architecture, I tried many changes in network architecture, including the following:

* 3 convolution layers instead of two
* connecting the outputs of the last two convolutions to the fully connected output layers, rather than just the last convolution
* changing dropout keep_prob between 0.5 and 1.0
* changing the depth of each convolutional layer (e.g. double, half)
* changing the number of fully connected units (e.g. double, half)
* removing the max_pool layers
* changing the kernel size of the convolutional layers to 3,3 instead of 5,5
* changing the padding from VALID to SAME

With each experiment, I chose the version that performed best. This results in a local maximum of model architecture changes.

To help understand the results better, I also added tensorboard integration and logged summaries of the convolutional variables & the validation & training error/loss.

My final model consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64				|
| RELU					|												|
| Dropout					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x128	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128			|
| RELU					|												|
| Dropout					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 3x3x256	|
| Max pooling	      	| 2x2 stride,  outputs 2x2x256			|
| RELU					|												|
| Dropout					|												|
| Fully connected (From layers 2 and 3)		| inputs: 4224, outputs: 384       									|
| RELU					|												|
| Fully connected		| inputs: 384, outputs: 128       									|
| RELU					|												|
| Fully connected		| inputs: 128, outputs: 43 (num_classes)       									|
| Softmax				|         		predictions							|


![network comparison][network]

#### Model Training

Hyperparameters:

|Hyperparemter|Value|Notes|
|:-----------:|:---:|:---:|
|Batch size   |2048 |This is close to the maximum that fits on my GPU|
|Epochs       |20 | |
|Learning Rate|0.0005|Decreasing this further didn't seem to increase the accuracy|
|keep_prob|0.6|only for training|

I used the Adam optimizer.


#### Approach to improving my model

My final model results were:
* training set accuracy of 0.94
* validation set accuracy of 0.98
* test set accuracy of 0.96

My first architecture was basic LeNet, which achieved:
* Train Accuracy: 1.00
* Valid Accuracy: 0.86
* Test Accuracy: 0.84

Clearly this was overfitting, so I tried adding dropout, and a conversion to grayscale.

My middle architecture (2 conv layers, grayscale, normalization, no augmentation) produced
* Train Accuracy: 1.00
* Valid Accuracy: 0.96
* Test Accuracy: 0.94

Still overfitting.

To further improve, I added network architecture features described in [sermanet-ijcnn-11.pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I implemented my own jitter for data augmentation, as well as a 3 layer network with the outputs from both layers 2 and 3 connected to the fully connected layer.

I tried training for as long as 10000 epochs, and with tweaks to other parameters. The parameters described above are what I found to perform best.

### Test a Model on New Images

Here are 9 German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5]![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9]

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of way at the next intersection      		| Right of way at the next intersection   									|
| Children Crossing     			| Road narrows on the right 										|
| Speed Limit (60km/h)					| Speed Limit (60km/h)											|
| General Caution	      		| General Caution					 				|
| Go straight or right | Go straight or right |
| Priority road | Priority road |
| Speed limit (20km/h) |Speed limit (20km/h) |
| Yield | Yield |
| Roundabout mandatory | Roundabout mandatory


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 89%. With so little data it's hard to compare accuracy directly to the validation & test sets, but it's within the ballpark.

The model is really certain about the predictions it makes:

~~~~
Right-of-way at the next intersection:1.000
Beware of ice/snow:0.000
Slippery road:0.000
Double curve:0.000
Road work:0.000

Road narrows on the right:0.990
Children crossing:0.010
Pedestrians:0.000
Road work:0.000
Slippery road:0.000

Speed limit (60km/h):1.000
Speed limit (20km/h):0.000
Speed limit (80km/h):0.000
Speed limit (50km/h):0.000
Speed limit (30km/h):0.000

General caution:1.000
Traffic signals:0.000
Pedestrians:0.000
Road narrows on the right:0.000
Wild animals crossing:0.000

Go straight or right:1.000
Ahead only:0.000
Turn right ahead:0.000
Go straight or left:0.000
Turn left ahead:0.000

Priority road:1.000
Roundabout mandatory:0.000
No entry:0.000
Keep left:0.000
Stop:0.000

Speed limit (20km/h):1.000
Speed limit (30km/h):0.000
Speed limit (80km/h):0.000
Speed limit (120km/h):0.000
Speed limit (70km/h):0.000

Yield:1.000
Priority road:0.000
No vehicles:0.000
No passing:0.000
Keep right:0.000

Roundabout mandatory:1.000
Priority road:0.000
Speed limit (100km/h):0.000
Turn right ahead:0.000
Speed limit (30km/h):0.000
~~~~

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

This network seems to focus heavily on edges. Each of the feature maps specializes in a different type of edge, e.g. diagonal, horizontal, vertical. low frequency, high frequency, and more.

The second layer simplifies this into a lower resolution approximation. It's also able to pick up on things being inside other things, e.g. FeatureMap 99 and FeatureMap 103.

Conv1 Feature Maps
![alt text][featuremaps-conv1]

Conv2 Feature Maps
![alt text][featuremaps-conv2]
