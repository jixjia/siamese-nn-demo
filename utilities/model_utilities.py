# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import random

def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    triplet = []

    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    # numClasses = len(np.unique(labels))
    # idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    # loop over all images
    for idx in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idx]
        currentLabel = labels[idx]

        # randomly pick an image that belongs to the *same* class
        # label
        idxPos = [i for i,elem in enumerate(labels) if elem == currentLabel]
        posImage = images[random.choice(idxPos)]

        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        idxNeg = [i for i,elem in enumerate(labels) if elem != currentLabel]
        negImage = images[random.choice(idxNeg)]

        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

        triplet.append((currentImage, posImage, negImage))

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels)), triplet


def make_pairs_apn(anchors, anchorLabels, positives, positiveLabels):
    
    pairImages = []
    pairLabels = []
    triplet = []

    # loop over all images
    for idx in range(len(anchors)):

        anchor = anchors[idx]
        anchorLabel = anchorLabels[idx]
        
        idxPos = positiveLabels.index(anchorLabel)
        positive = positives[idxPos]

        idxNeg = [i for i,elem in enumerate(positiveLabels) if elem != anchorLabel]
        negative = positives[random.choice(idxNeg)]

        # generate a,p,n pairs
        pairImages.append([anchor, positive])
        pairLabels.append([1])

        pairImages.append([anchor, negative])
        pairLabels.append([0])

        triplet.append((anchor, positive, negative))

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels)), triplet


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)

    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)