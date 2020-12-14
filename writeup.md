# **Traffic Sign Recognition** 
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_plot.png "Number of training images"
[image2]: ./examples/data_aug.png "Data augmentation"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/signs.png "Traffic Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

> 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kmiya/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

> 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I just used `numpy.ndarray.shape` and `len()` to show the basic summary of the dataset.

```
n_train      = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test       = X_test.shape[0]
image_shape  = X_train.shape[1:]
n_classes    = len(set(test["labels"]))

# Number of training examples = 34799
# Number of testing examples = 12630
# Image data shape = (32, 32, 3)
# Number of classes = 43
```

> 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. The training data looks _skew_. It may affect the performance of the trained model.

![alt text][image1]

### Design and Test a Model Architecture

> 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Data augmentation is known to be a good practice to prevent over fitting. I used the following function to augment training data. The function slides, rotates, scales an input image and adds values randomly. The ranges of values `inp`, `scale`, and `rot` were referred to [Sermanet & LeCun, 2011](https://doi.org/10.1109/IJCNN.2011.6033589). In my case, adding random values to the input image greatly improved accuracy.
```python
import random

def data_augment(img):
    # ref: Sermanet & LeCun, 2011. https://doi.org/10.1109/IJCNN.2011.6033589
    result = np.copy(img)
    inp = random.uniform(-2, 2) + 16
    scale = random.uniform(.9, 1.1)
    rot = random.uniform(-15, +15)
    lit = random.randint(-30, +30)
    result = np.clip(result + lit, 0, 255)
    mat = cv2.getRotationMatrix2D((inp, inp), rot, scale)
    return cv2.warpAffine(result, mat, (32, 32), borderMode=cv2.BORDER_REPLICATE)
```
The followings are examples of the data augmentation.
![Data augmentation][image2]

> 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is almost the same as the LeNet-5, but using He initialization technique ([He et al., 2015](https://doi.org/10.1109/ICCV.2015.123)) to stabilize the training process.
```python
def he_initialize(n_nodes, shape, seed=0):
    mu = 0
    sigma = sqrt(2. / n_nodes)
    return tf.truncated_normal(shape=shape, mean = mu, stddev = sigma, seed=seed)
```


| Layer         		| Description   	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, outputs 28x28x6     				|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6     				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16					|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x6     				|
| Fully connected		| outputs 120      								|
| Fully connected		| outputs 84       								|
| Fully connected		| outputs 43       								|
| Softmax				|           									|
|						|												|
 


> 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I decided the hyperparameters by experiments.

| Hyper Parameter  		| Values        	        					| 
|:---------------------:|:---------------------------------------------:|
| Optimizer             | Adam                                          |
| Learning rate			| 0.001											| 
| Batch Size       		| 256                  							| 
| Number of epochs     	| 101                            				|
|                       |                                               |

> 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
```
* training set accuracy   : 0.998
* validation set accuracy : 0.946
* test set accuracy       : 0.927
```

1. At first, I used the LeNet-5 solution of the lab with no changes, but the accuracy of the model was about ~90%.
2. Next, I increased the number of filters (channels) of the CNN layer 1 and 2, but it failed, difficult to converge the validation loss.
3. Then I tried to apply He initialization to LeNet-5 mentioned above. It worked, the accuracy became ~93%.
4. Finally, I added augmented training data. I think it is a most important point to improve the performance of the model. The accuracy became ~95%.

To improve my model more, the following famous techniques would work well: ResNet, EfficientNet, Dropout, Batch Norm, AugMix, etc.

### Test a Model on New Images

> 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on [Wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany), which are resized to 32x32 pixels:

![alt text][image4]

The signs mean:
- Speed limit (60km/h)
- No passing for vehicles over 3.5 metric tons
- Stop
- No vehicles
- Double curve
- Bumpy road
- End of no passing
- End of no passing by vehicles over 3.5 metric tons

The first six signs might be difficult to classify because the red color dominates the images and they has similar shapes each other. The last two images also might be difficult to classify because they are almost the same signs.

> 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

```
0 correct: (3) Speed limit (60km/h)
  predict: (15) No vehicles, prob: 0.2886759638786316
1 correct: (10) No passing for vehicles over 3.5 metric tons
  predict: (15) No vehicles, prob: 0.21152722835540771
2 correct: (14) Stop
  predict: (15) No vehicles, prob: 0.23726747930049896
3 correct: (15) No vehicles
  predict: (15) No vehicles, prob: 0.1577601283788681
4 correct: (21) Double curve
  predict: (41) End of no passing, prob: 0.22841554880142212
5 correct: (22) Bumpy road
  predict: (41) End of no passing, prob: 0.1777673363685608
6 correct: (41) End of no passing
  predict: (15) No vehicles, prob: 0.1662244200706482
7 correct: (42) End of no passing by vehicles over 3.5 metric tons
  predict: (15) No vehicles, prob: 0.16806964576244354
```

```
Top-1 accuracy: 12.5 % (1/8)
```


My model tends to predict some reddish signs as _No Vehicles_. It would be understandable because the six of eight images has red edges.

> 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top-5 softmax probabilities are as follows:
```
correct: 3
predict 15: 28.9 %
predict 41: 22.7 %
predict  3: 13.5 %
predict 29: 12.9 %
predict  2: 9.95 %

correct: 10
predict 15: 21.2 %
predict 41: 18.0 %
predict  3: 8.07 %
predict  4: 7.75 %
predict 33: 6.85 %

correct: 14
predict 15: 23.7 %
predict 41: 21.6 %
predict  4: 8.3 %
predict 17: 7.16 %
predict 13: 6.33 %

correct: 15
predict 15: 15.8 %
predict 41: 15.4 %
predict  4: 8.29 %
predict  9: 5.72 %
predict 29: 5.34 %

correct: 21
predict 41: 22.8 %
predict 15: 17.0 %
predict 28: 12.5 %
predict 29: 12.4 %
predict 33: 9.8 %

correct: 22
predict 41: 17.8 %
predict 15: 14.6 %
predict 29: 11.1 %
predict  4: 10.4 %
predict 33: 9.41 %

correct: 41
predict 15: 16.6 %
predict 41: 13.8 %
predict 33: 10.2 %
predict  4: 8.16 %
predict  2: 6.74 %

correct: 42
predict 15: 16.8 %
predict 41: 16.6 %
predict 33: 13.4 %
predict 29: 11.2 %
predict  3: 8.36 %
```
The top-5 accuracy is:
```
Top-5 accuracy: 37.5 % (3/8)
```
To see the result, I could not say that the model is good to classify traffic signs. To improve the model, for example, decreasing the size of filters from 5x5 to 3x3 would seem to work because 5x5 filters might lose the details of the signs. Recent data augmentation techniques like AugMix might be also promising. The techniques are known to be good at improving the robustness of the model.