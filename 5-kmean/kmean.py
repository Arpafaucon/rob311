#!/usr/bin/python3
# coding: utf8

"""
K-Mean classification of handwritten digits
The K-Mean algorithm is run with an arbitrary number of clusters.
A label association step is then performed: for each predicted cluster, we determine the most present actual class, and name the cluster after it.
This technique handles better the different shapes of a same digits (especially if they are close from another digit).
the variable of interest is the number of clusters
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.cluster
import sklearn.metrics
import plot_confusion_matrix

# black and white color palette
GRAYSCALE_CMAP = sns.dark_palette('white', 16)

# constants 
DIGIT_SHAPE = (8,8)
NUM_FEATURES = 64
ACTUAL_CLASS = NUM_FEATURES



# data inputs
data_training = pd.read_csv('data/optdigits.tra', sep=',', header=None)
data_testing = pd.read_csv('data/optdigits.tes', sep=',', header=None)


def plot_datapoint(datapoint):
    """
    Utility. Plot one digit datapoint
    
    Args:
        datapoint (pd.Dataframe|np.array): feature vector of the datapoint, with length >= NUM_FEATURES
    """
    print(datapoint)
    if isinstance(datapoint, pd.DataFrame):
        pixels = datapoint.values[0:NUM_FEATURES]
    else:
        pixels = datapoint[0:NUM_FEATURES]
    print(pixels)
    pixels_2d = pixels.reshape(DIGIT_SHAPE)
    sns.heatmap(pixels_2d, cmap=GRAYSCALE_CMAP)
    plt.show()

def compute_label_reassociation(training, raw_labels, num_kmean_classes):
    """
    Associate the predicted labels of the training dataset with the real digit number.
    
    Args:
        training (pd.Dataframe): training dataset
        raw_labels (list[int]): predicted labels for each datapoint
        num_kmean_classes (int): number of kmean classes used
    
    Returns:
        (list[int], (int)->int): label association list, function mapping a raw label index to the actual digit label
    """
    label_votes = np.zeros((num_kmean_classes, 10), dtype=np.int32)
    for i_pt in range(len(training)):
        real_label = training.values[i_pt, ACTUAL_CLASS]
        predicted_label = raw_labels[i_pt]
        label_votes[predicted_label, real_label] +=1

    label_association = np.zeros(num_kmean_classes, dtype=np.int32)
    for i_raw_label in range(num_kmean_classes):
        max_voted = np.argmax(label_votes[i_raw_label])
        label_association[i_raw_label] = max_voted

    def association_function(
            i_raw_label):
        """
        mapping function
        
        Args:
            i_raw_label (int): index of the raw predicted label
        
        Returns:
            int: actual digit label
        """
        return label_association[i_raw_label]

    return label_association, association_function


def scikit_kmeans(training, testing, num_clusters):
    """
    Core k-mean learning procedure
    
    Args:
        training (pd.Dataframe): training dataset
        testing (pd.Dataframe): testing dataset
        num_clusters (int): number of cluster to use
    
    Returns:
        list[int]: list of label mapping each datapoint of the testing dataset to a digit label
    """
    unsup_training = training.values[:, 0:NUM_FEATURES]
    unsup_testing = testing.values[:, 0:NUM_FEATURES]
    # training
    kmean = sklearn.cluster.KMeans(n_clusters=num_clusters, init='k-means++')
    kmean.fit(unsup_training)
    # prediction
    training_predictions = kmean.predict(unsup_training)
    testing_predictions = kmean.predict(unsup_testing)
    # reassociation
    label_list, label_reassoc_fun = compute_label_reassociation(data_training, training_predictions, num_clusters)
    better_labels = list(map(label_reassoc_fun, testing_predictions))
    return better_labels

def score_results(testing_dataset, labels, num_clusters, save_fig=False):
    """
    Learning evaluation function.
    Computes the confusion matrix and the average score
    
    Args:
        testing_dataset (pd.Dataframe): testing dataset
        labels (list[int]): labels for the testing dataset
        num_clusters (int): number of clusters used (display purpose only)
        save_fig (bool, optional): Defaults to False. Whether to save the confusion matrix in a file
    
    Returns:
        np.array[10,10],float: confusion matrix, overall score
    """
    conf_matrix = sklearn.metrics.confusion_matrix(testing_dataset.values[:,NUM_FEATURES], labels)
    total_pts = len(testing_dataset)
    correct_pts = np.trace(conf_matrix)
    score = correct_pts/total_pts

    if save_fig:
        fig_filename = 'fig/confusion_{}_clusters'.format(num_clusters, int(score*1000))
        fig_title = '{} clusters (score: {:.2f}%)'.format(num_clusters, score)
        plot_confusion_matrix.cm_analysis(testing_dataset.values[:,NUM_FEATURES], labels, [i for i in range(10)],  filename=fig_filename, title=fig_title)
    return conf_matrix, score


def plot_mean_digits(dataset, labels, num_clusters):
    """
    Display the average digit image of each kmean class.
    
    Args:
        dataset (pd.Dataframe): dataset
        labels (list[int]): predicted labels for the above dataset
    """
    # compute sum image for each class
    sum_digit = np.zeros((10,NUM_FEATURES))
    num_digit = np.zeros(10, dtype=np.int32)
    pixels = dataset.values[:,0:NUM_FEATURES]
    for digit_pixels, pred_label in zip(pixels, labels):
        sum_digit[pred_label,:] += digit_pixels
        num_digit[pred_label] +=1
    
    # display average image
    fig = plt.figure()
    axes = fig.subplots(2,5).reshape(-1)
    for label in range(10):
        mean_digit = sum_digit[label,:] / num_digit[label]
        pixels_2d = mean_digit.reshape(DIGIT_SHAPE)
        # ax = plt.subplot(2,5,1+label)
        sns.heatmap(pixels_2d, ax=axes[label], vmin=0, vmax=16, cbar=False, cmap=GRAYSCALE_CMAP)
        axes[label].set_axis_off()
    fig.suptitle('Average digit representation')
    # fig.show()
    fig.savefig('fig/digits_{}_clusters'.format(num_clusters))
    plt.show()


def compare_num_clusters():
    """
    Compare the overall score for various cluster sizes
    """

    def cluster_size_gen():
        """
        generates the list of sizes to test
        """
        yield from range(5,10,2)
        yield from range(10,16,1)
        yield from range(16,41,4)
    
    ncluster_list = list(cluster_size_gen())
    score_list = []
    print('starting comparison. \nList of cluster sizes : {}'.format(ncluster_list))
    for n_clusters in ncluster_list:
        predicted_labels = scikit_kmeans(data_training, data_testing, n_clusters)
        _, predict_score = score_results(data_testing, predicted_labels, n_clusters, save_fig=False)
        print('{} clusters: score = {:2f}'.format(n_clusters, predict_score))
        score_list.append(predict_score)
    # plt.clf()
    
    plt.plot(ncluster_list, score_list)
    plt.xlabel('clusters')
    plt.ylabel('score')
    plt.title('Evolution of the classification score with the number of K-Means clusters')
    plt.savefig('fig/score_comparison')
    plt.show()

def analysis(n_clusters=12):
    """
    More detailede analysis of the K-Mean results for a given cluster size

    Args:
        n_clusters (int, optional): Defaults to 12. number of clusters
    """
    # confusion matrix
    predicted_labels = scikit_kmeans(data_training, data_testing, n_clusters)
    conf_matrix, predict_score = score_results(data_testing, predicted_labels, n_clusters, save_fig=True)
    print('Results for n_cluster={}:'.format(n_clusters))
    print('score : {:.2f}\nConfusion matrix:\n{}'.format(predict_score, conf_matrix))
    plt.show()
    # average class digit representation
    plot_mean_digits(data_testing, predicted_labels, n_clusters)

if __name__ == '__main__':
    plt.ion()
    compare_num_clusters()
    analysis(12)
    plt.waitforbuttonpress()