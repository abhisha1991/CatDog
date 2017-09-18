This is a simple example of a convolutional NN binary classifier that can be extended as a n-ary classifier just as easily.
This reads the kaggle dataset (https://www.kaggle.com/c/dogs-vs-cats) to classify the test images being dogs or cats. 

There are 6 convolutional layers in the model of varying hidden layer sizes (32 to 128) with a softmax layer at the end to classify output probabilities.

The model trains on approx 9000 training images and achieves an accuracy of 75%
![alt text](https://github.com/aus2101/CatDog/blob/master/log/bestmodel.PNG)


Below is a sample from the classification on the test set images. As you can see, the model was able to accurately classify images to a large extent.
![alt text](https://github.com/aus2101/CatDog/blob/master/log/catdog.png)

## Dependencies:
1. cv2 (opencv for resizing images)
2. numpy
3. Tqdm (for visualization)
4. Pyplot/Matplotlib

## Deep learning Dependencies:
1. tflearn
2. tensorflow

This works well with Python 3.

## Basic Usage
Simply clone the repo and run **'python CatOrDog.py'**
