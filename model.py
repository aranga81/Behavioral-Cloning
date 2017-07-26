#  IMPORTS...!!!!
import numpy as np
import csv
import cv2
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D


#######################################################################################

# function to read in the training/validation data from the logs
lines = []

def load_csvdata(log_file):

    with open(log_file) as file:
        reader = csv.reader(file) # Read csv log file
        for line in reader:
            lines.append(line)

    images = []
    steering_angles = []
    steering_corr = 0.20

    for line in lines[1:20]:
        for i in range(3):
            path = line[i]
            token = path.split('/')
            filename = token[-1]
            images_path = "./SimulationData/IMG/" + filename
            # print(filename)

            image = cv2.imread(images_path)
            images.append(image)

        steering_angle = float(line[3])
        steering_angles.append(steering_angle)
        steering_angles.append(steering_angle + steering_corr)
        steering_angles.append(steering_angle - steering_corr)
    return images, steering_angles

#######################################################################################

def data_distribution(angles):

    num_bins = 15
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()


#######################################################################################
# Data Pre Processing Function

def preprocess(img):

    # original shape: 160x320x3, Cropping operation
    new_img = img[50:140, :, :]

    # gaussian blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)

    # scale to 66x200x3 (nVidia model)
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)

    # convert to YUV color space (nVidia model)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

#######################################################################################

def plot_data(images):
    # idx = random.randint(0, len(images) - 1)
    pr_im = preprocess(images[1])
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(pr_im, cv2.COLOR_BGR2RGB))
    plt.show()

#######################################################################################

def data_augmentation(images, angles):
    aug_images = []
    aug_steeringangles = []

    for image, steering_angle in zip(images, angles):
        aug_images.append(image)
        aug_steeringangles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_steeringangle = float(steering_angle)*-1.0
        aug_images.append(flipped_image)
        aug_steeringangles.append(flipped_steeringangle)

    x_train_aug = np.array(aug_images)
    y_train_aug = np.array(aug_steeringangles)

    return x_train_aug, y_train_aug


#######################################################################################

def data_preprocessing(images, angles):
    aug_process_images = []
    aug_steering_angle = []

    #  Pre processing entire augmented data set
    for image, steering_angle in zip(images, angles):
        processed_image = preprocess(image)
        aug_process_images.append(processed_image)
        aug_steering_angle.append(steering_angle)

    x_processed = np.array(aug_process_images)
    y_processed = np.array(aug_steering_angle)
    return x_processed, y_processed


#######################################################################################
#  DEFINE SEQUENTIAL MODEL

model = Sequential()
#  first
model.add(Lambda(lambda x: x/127.5 - 1.0,  input_shape=(66, 200, 3)))
# second layer
model.add(Conv2D(24, (5, 5), strides=(2, 2), data_format="channels_last", padding="valid", activation="elu", W_regularizer=l2(0.001)))
print(model.output_shape)

# third Layer
model.add(Conv2D(36, (5, 5), strides=(2, 2), data_format="channels_last", padding="valid", activation="elu",  W_regularizer=l2(0.001)))
print(model.output_shape)
# fourth Layer
model.add(Conv2D(48, (5, 5), strides=(2, 2), data_format="channels_last", padding="valid", activation="elu",  W_regularizer=l2(0.001)))
print(model.output_shape)
# fifth Layer
model.add(Conv2D(64, (3, 3), data_format="channels_last", padding="valid", activation="elu",  W_regularizer=l2(0.001)))
print(model.output_shape)
# sixth Layer
model.add(Conv2D(64, (3, 3), data_format="channels_last", padding="valid", activation="elu",  W_regularizer=l2(0.001)))
print(model.output_shape)

# Flatten layer
model.add(Flatten())
print(model.output_shape)

# seventh layer
model.add(Dense(100, activation='elu',  W_regularizer=l2(0.001)))

# eighth layer
model.add(Dense(50, activation='elu',  W_regularizer=l2(0.001)))

# ninth layer
model.add(Dense(10, activation='elu',  W_regularizer=l2(0.001)))

# Output layer
model.add(Dense(1))


##############################################################
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

x_train, y_train = load_csvdata('./SimulationData/driving_log.csv')
print(np.shape(x_train))
print(len(y_train))

plot_data(x_train)

data_distribution(y_train)

x_train, y_train = shuffle(x_train, y_train, random_state=14)
X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=14)

print(np.shape(X_train))
print(np.shape(X_validation))
print(len(y_train))
print(len(y_validation))


X_train_aug, y_train_aug = data_augmentation(X_train, y_train)
print(np.shape(X_train_aug))
print(len(y_train_aug))
data_distribution(y_train_aug)

X_train_aug, y_train_aug = data_preprocessing(X_train_aug, y_train_aug)
X_validation, y_validation = data_preprocessing(X_validation, y_validation)

print(np.shape(X_train_aug))
print(len(y_train_aug))

print(np.shape(X_validation))
print(len(y_validation))
plot_data(X_train_aug)

##############################################################
model.compile(loss='mse', optimizer='adam')

model.fit(X_train_aug, y_train_aug, batch_size=100, shuffle=True, nb_epoch=20, validation_data=(X_validation, y_validation))

print(model.summary())
print('saving model weights..!!')

model.save('model.h5')
