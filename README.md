# Behavioral-Cloning
Project pipeline is used to train and validate Nvidia's end-to-end model using Drive Simulator

Project goals:
•	Use the simulator to collect data of good driving behavior (Simulator From UDACITY self driving car nano degree)
•	Build, a convolution neural network in Keras that predicts steering angles from images
•	Train and validate the model with a training and validation set
•	Test that the model successfully drives around track one without leaving the road

The project package includes:
-Model.py – Code to train the network and save the weights
-Drive.py – used to drive the car autonomously in the simulation (modification – added data pre-processing function for input images before using model weights).
-Video1 –track1 autonomous driving.
-Video2 – track 2 autonomous driving.



## Data Collection:

Used the Simulator provided by [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3) was used to generate the data for two different tracks using center, left and right camera. The image below shows the center/left/right camera view for both the tracks.

![Data visualization](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/data_visualization.JPG)

The generated data from the simulator consists about 27663 samples from all the three camera views used later towards training and validation sets.
data was generated from following drive cycles - best way to generalize your model...!!! (change that I had to include to make sure the model performs well).
-	Drive 1: track 1 normal
-	Drive 2: Track 1 – in reverse direction (to make sure we have equal left and right turns as track one normal direction is biased more     with left only turns)
-	Drive 3: Track 2 – normal
-	Drive 4: Track 2 – reverse direction

## Data Augmentation:

Before I augment any data I did shuffle the data and also split 20% of the data into a validation set using train_test_split from sklearn.model_selection. This way I was only going to augment my training data and not the validation set.

Best way to generate additional fake data from the original set - by flipping, changing brightness / filtering, warping etc. This is a step to reduce any overfitting and also helps generalize the model a bit. I implemented data augmentation by flipping the images about the vertical axis and likewise multiplying the steering angles by -1 (so we have to steering in the opposite direction). This operation to augment the images will make sure we have equal distribution of steering left and right data. 

## Data Processing:
I cropped the images from all the cameras by removing the top 20 pixels and also the bottom 50 pixels. The reason for cropping the top 20 pixels is mainly because it consists of background noise from trees and sky. The bottom 50 pixels are covered with the hood of the vehicle and not much feature left to extract. 

![Cropped Images](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/cropping.png)


I used a gaussian filter in the processing stage of filter size 3x3 to filter any additional noises in the images. This step was considered after looking closely at some of the noisy images from left and center camera especially on track 2.

![Gaussian Filter](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/gaussianblur.png)

## Training Time....!!!

The network architecture used for this project was from [Nvidia's end-to-end learning model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The input images had to be resized to 200x66x3 and split into YUV plane. The conversion and resizing step's are included in the data processing functionality.

![resize](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/yuv.png)

##### All the training and validation sets go through the pre-processing stage for training & model validation. Also the function is included in drive.py code for the new input images while in autonomous drive mode.

The final model architecture that was used for this project is from the NVidia’s published paper on end-to-end learning. The architecture contains a total of 9 layers with first 5 convolutional layers followed by fully connect ones.

![network architecture](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/model.png)

![network architecture](https://raw.github.com/aranga81/Behavioral-Cloning/master/output_images/parameters.png)

The NVidia paper does not mention about any strategies to reduce overfitting and use of any activation functions in the layers. But for this final model I used “ELU” activation function on each convolutional and fully connected layer. 
To reduce overfitting I choose to include l2 normalization (0.001) in each of the convolution and fully connected layers.
Optimizer – Adam 
Number of Epochs: 20 (Going over 20 / 23 I see that I was running into overfitting issues)
Batch Size: 100
I choose to use Keras ‘fit’ method to train the images. (Could implement the fit_generator method and use the generator on each batch).

# Watch the Autonomous drive mode videos (Video1 & Video2)....!!!!






