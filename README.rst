================
Introduction
================
This project aims to find best hyperparameters for neural network, to create model, which task is to recognize emotions based on facial expressions.

Model is trained on the FER-2013 dataset. Images are converted for easier use.

================
Dependencies
================
Python 3, OpenCV, Tensorflow, Hyperopt

================
Additional Information
================
For purpose of optimization of hyperparameters, Parzen Tree estymator was used. Searched hyperparameters space consisted of number of neurons in layer and value of dropout part. Existence of 3 layers of this model also belongs to the space of hyperparameters searched.
