#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/roadworks.jpg "Traffic Sign 6"
[image4]: ./examples/bumpy.jpg "Traffic Sign 1"
[image5]: ./examples/curve_left.jpg "Traffic Sign 2"
[image6]: ./examples/curve_right.jpg "Traffic Sign 3"
[image7]: ./examples/ice.png "Traffic Sign 4"
[image8]: ./examples/roadnarrows.png "Traffic Sign 5"


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/hsuvarna/TrafficSigns/blob/master/Traffic_Sign_Classifier.ipynb)


###Data Set Summary & Exploration

I used the standard python length utlities provided on top of np arrays to calculate summary statistics of the traffic
signs data set:

* The size of training set is : 34799
* The size of the validation set is : 4410
* The size of test set is : 12630
* The shape of a traffic sign image is : 32x32x3 (RGB)
* The number of unique classes/labels in the data set is : 43

####2. Include an exploratory visualization of the dataset.

The things about the data visualization that stuck my mind are 
* how many different shapes are there in the problem
* how many colors are there in input
* is text present in input data
* are deformed (equivalent of italics, bold in text) objects present
* which objects are close
* which objects are 180 degrees i.e. opposite i.e. reflective
* which objects are magnified version of other objects

More about the content so that convolution windows can be guessed. But ran out of time/coding knowledge to depict.


###Design and Test a Model Architecture

I have tried the following first to process images
1. Tried converting the images to grayscale and used the normalization technique to clamp values between 0 and 1. My training accuracy was abysmal of <10%. This could be because of bugs in my normalization code.
2. I also tried RGB normalization by computing (R/R+G+B)\*255 method. It was very slow as I was operating at pixel level.
3. Then I tried the simple linear alzebra way of dividing by 255 and subtracting 0.5 to clamp values [-0.5 to 0.5]. This was fast and worked.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| Normalized to [-0.5 , 0.5]|
| Convolution 5x5x3x6     	| outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5x6x16	    | 10x10x16 image     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Flattened      | 5*5*16=400 |
| Fully connected 400		|    Output 120    									|
| RELU				|         									|
| Fully connected 120		|    Output 84    									|
| RELU				|         									|
| Fully connected 84		|    Output 43   									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the above LENET neural net of 5 levels. The batch size was 100, epochs 20 and learning rate of 0.001. All the labels were one hot encoded. The convergence to minimum error was attained using the AdamOptimizer function. I n the class we only learned about Gradientdescent optimizer. The optimizer per stack echange link https://stats.stackexchange.com/questions/184448/difference-between-gradientdescentoptimizer-and-adamoptimizer-tensorflow is faster at the expense of more computation and memory.

After the preprocessing gave me fits of accuracies around < 10%, I fixed the preprocessing to normalization to [-0.5, 0.5] for all channels of the image. This immediately bumped up my accuracies into high 70s. Then I played with the epochs, learning rate and batch size.

| Epochs         		|    Batch size	        					| Learning rate      | Accuracies 
|:---------------------:|:---------------------------------------------:|:---------------------:|:---------------------:|
| 50         		| 100   							| 0.0001     | Validation = 0.84, Test=0.25 |
| 50         		| 100   							| 0.01       | Validation = 0.9, Test=0.88 |
| 10         		| 50  							  | 0.00001     | Validation = 0.11, Test=0.0.07 |
| 10         		| 150   							| 0.0001     | Validation = 0.74, Test=0.73 |
| 10         		| 100   							| 0.01     | Validation = 0.90, Test=0.889 |
| 10         		| 150   							| 0.0001     | Validation = 0.74, Test=0.73 |
| 10         		| 100   							| 0.01     | Validation = 0.90, Test=0.89 |
| 10         		| 100   							| 0.01     | Validation = 0.902, Test=0.889 |
| 20         		| 100   							| 0.001     | Validation = 0.936, Test=0.922 |


I settled for the last row. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

I just experimented with various epochs, learning rates and batch sizes. This combination is working to give validation accuracy > 0.93. 

I also tried dropouts on conv1 and conv2 (1st 2 layers) with probability 0.5. But that worsened the accuracies to 0.6-0.8. I abandoned the dropout experiment.

The current neural net model seems like working good. Basically the input problem we have is characterized by pretty naroow set of image content. i.e. all the german traffic signs are about traingles (blue and red), circles (blue and red), arrows and curves. The neural net convolutions in the 1st two layers are enough to learn about these basic shapes. This problem is slightle extended version of the mnist digits where in only 10 digits are there and all are pretty well defined through curves.
However, this may not be enough for objects recognistion, patterns.


My final model results were:
* training set accuracy of 0.936
* validation set accuracy of 0.936
* test set accuracy of ? 0.922

If an iterative approach was chosen:
No. I started with LENET and it worked ok.

If a well known architecture was chosen:
* What architecture was chosen?  

LENET

* Why did you believe it would be relevant to the traffic sign application? 

Based on the good results it gave for mnist. The traffic isgns are also limited shapes like mnist digits.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
If the test images are much like training images i.e. cropped to the center, then the test results were good. So if we keep same model of preprocessing, cropping, shape of training/test images, we can get good results.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The slippery road image ![alt text][image7] might be difficult to classify because it has a tree kind of shape which is similar to another road sign of trees. 

The bumpy sign image ![alt text][image4] is also hard as it has a red traiangle and two curves inside it similar to some other traffic signs like roadworks.

The curve left and double curve left also are similar images with slight shapes.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									| 
| wild animal sign     			| wild animals sign 										|
| slippery road					| double curve										|
| curve left	      		| curve left				 				|
| curve right			| curve right     							|
| no curve ahead			| no curve ahead    							|
| bike crossing		|     bike crossing							|
| road narrows		| road narrows      							|
| road works			| road works     							|
| bumpy road			| bumpy road     							|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

examples/curve_left.jpg
examples/curve_right.jpg
examples/road_narrows.jpg
examples/ice.jpg
examples/bumpy.jpg
examples/roadworks.jpg
examples/wild.jpg
examples/noveh.jpg
examples/bike.jpg
examples/stop.jpg
image shape is (32, 32, 3)
length 10

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9984503          			| curve_left 									| 


For the second image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000        			| curve_right 									|


For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9991807         			| curve_right 									| 
| 0.0008193        			| slippery									| 

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9683495         			| children crossing (wrong)									| 
| 0.0316506       			| slippery									| 

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         			| bumpy									| 
| 0.0         			| pedestrians						| 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


