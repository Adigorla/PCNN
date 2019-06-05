# PCNN

First, please install any required packages, and make sure you are using *Tensorflow 2.0 alpha*

In tf_prac1.py, we have implemented a simple convolutional neural network composed of 32 conv. filters with 3x3 kernels followed by two dense layers. This is the "control" to which we are comparing our PST neural network.

In pst_m1.py, we have implemented a neural network that has a single PST layer followed by two dense layers, trained with gradient descent.

To run either of these networks for 10 epochs, download the required packages for each file and run "python3 ./pst_m1.py" or "python3 ./tf_prac1.py" to run the networks. The training and test loss/accuracy for each epoch will be printed to standard output, as well as the training and testing times. Please note that each epoch takes a little over 2 minutes to run for pst_m1.py, so give it some time. Each epoch takes approx. 1 min for the regular CNN.
