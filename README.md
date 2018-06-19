
## Segmantic Segmentation ##

Segmantic Segmentation is that task of asisgning meaning to part of an object. With modern deep learning techniques, this can be at the pixel level, where each pixel is assigned to a target class. In this project, I have trained a fully convolutional neural network to identify a target person from images produced by a quadcopter simulator.

### Network Architecture 

A FCN architecture is comprised of an encoder and decoder. 

The encoder portion is a convolutional netwrok that reduces to a 1x1 convolutional layer, in contrast to a flat fully connected layer that would be used for image classfication. This difference has the effect of preserving spatial information from the image. Separable convolutions are used instead of the traditional convolutional layer. This technique reduces the number of parameters needed, thus increasing efficiency for the encoder network. They comprise of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and combines them into an output layer.

In addition to the use of separable convolutions, the encoder layer utilizes batch normalization. Instead of just normalizing the inputs ot the network, batch normalization normalizes the inputs to layers within the network using the mean and variance of the values in the current mini-batch. This technique allows the network to train faster by converging more quickly, allows higher learning rates, and provides a bit of regularization by adding noise to the network. Both separable convolutions and batch normalization are implemented via the separable_conv2d_batchnorm() function in Keras

The decoder portion of the network consists of upsampling layers, traditionally done with transposed convolutions. In this network, I am using bilinear upsampling, a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. After bilinear upsampling, the upsampled layer is concatenated with a previous layer with more spatial information, providing the same functionality as using skip connections. Finally, a separable convolution layer is added to better learn the spatial details from the previous layers. Together, these 3 steps (bilinear upsampling, concatenation, separable convolution) define 1 "decoder block".

Describe final model architecture.

Describe brute force trial and error process


### Hyperparameters

### Results

### Limitations and future improvements



## Running the project ##

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5



The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```


### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 
