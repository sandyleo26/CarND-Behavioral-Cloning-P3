#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia-model]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png

[//]: # (Links)
[nvidia]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
[oom]: https://github.com/aymericdamien/TensorFlow-Examples/issues/38

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I borrow the [model][nvidia-model-link] used by Nvidia for this project. The architecture is show below.

![network architecture][nvidia-model-img]

The model has 5 convolutional layers, some using sub-sampling with stride of 2 by 2, followed by 3 fully connected layers. The output is a single value which represent turning radius.

#### 2. Attempts to reduce overfitting in the model
TODO

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
TODO

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data
TODO

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple and progressively use more powerful networks.

Firstly, as David did in video lecture, just flatten the entire image, a fully connected layer and then output a single value. This is just to make sure environment is setup correctly and get some ideas of the potential difficult part. 

**The result**: loss is hugh, and car is hardly moving, constantly steering wheels. But at least we make the first step toward the goal.

Secondly, I replaced the model with a LeNet like network and increase training data by driving more laps. Besides, I also normalize the image in preprocessing stage. 

**The result**: both model size and loss decreas due to subsampling in the convolutional stage and car is steering smoothly, although will wander off in the first turn. I notice that there's still room for improving loss but increasing number of epochs can't guarantee that becasue the loss can suddenly increase dramatically, which usually leads to even worse model. This is a sign of underfitting.

Thirdly, main changes in this round:
* flipping images and using left and right camera views, resulting 6x data
* use the core nvidia pipeline
* use Udacity provided data set
* 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture basically copied the [pipeline][nvidia] with the following layers.

![network work architecture][nvidia-model]

However, I added two more convoluational layers in the begining, 8@66x200 with 7x7 and 16@66x200 with 5x5.

#### 3. Creation of the Training Set & Training Process

At first use the simulator to generate training data. I first recorded 2 laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane][center-lane]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn when driving off the center. These images show what a recovery looks like:

![recover 1][recover1]
![recover 2][recover2]
![recover 3][recover3]

To augment the data sat, I also flipped images and angles and use left and right camera images with adjusted angle(+/- 0.2)

However, at this phaze I realize data collection is very time consuming. First, you need to be really careful when recording. If bad driving is recoarded then to prevent it from being learned, I have to restart recoarding. This is tedious. And second, I have to upload it to AWS to training and it'll cost me hours to upload a few hundreds MB data. So I decided to use Udacity provided dataset.

The dataset has TODO...

To avoid [out of memory error][oom], I use generators to feed in data. The `batch_size` is also chosen to avoid memory issue. 

 The ideal number of epochs was TODO as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
