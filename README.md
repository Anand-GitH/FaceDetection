# FaceDetection
Haar Classifiers - Adaboost for efficient face detection


Open CV offers two types of pre-trained classifiers which are trained on multiple positive
and negative sample of images. We will use these classifiers and build algorithm for the sample
of dataset – FDDB.

Two pre-trained face detection classifiers by open cv :
1. Haar classifier
2. LBP – Local Binary Patterns

Haar classifier:

Haar classifier learns haar features mentioned below which are windows just like
convolutional kernels to learn image features. These windows move across image learning the
image features and each window when placed on image returns a single value for the whole
window by subtracting the sum of values of black portion from white portion. Larger images can
be scaled down to learn the features. As we learn non-essential features which will not addon to
classification accuracy so we use machine learning method Adaboost to build strong classifier
from the sequence of weak classifiers. Adaboost discards group features which does not add up to
classification accuracy.

LBP: Local Binary Patterns

For each pixel is compared to its neighbor pixels like 3*3 window with center being the
pixel of interest and if the value of neighbor pixel is greater than pixel of interest then set to 1
else 0 then summed up to get the value of LBP. Uses histogram of these blocks to create a
feature vector which contains features of interest.
