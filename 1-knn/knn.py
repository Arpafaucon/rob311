#/usr/bin/python3
# coding: utf-8

from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def dist_iris(a, b):
    """
    defines distances in the iris dataset
    
    Args:
        a (flower 4-uple): element
        b (flower 4-uple): element

    """
    dist = [a.iloc[i] - b.iloc[i] for i in range(4)]
    dist2 = [d**2 for d in dist]
    res = sum(dist2)
    return res

def knn_iris(flower, dataset, k):
    """
    knn search for the flower
    computes distance to all elements of the dataset and finds the winning class among the k nearest 
    
    Args:
        flower (flower 4-uple): 
        dataset (classified flower 5-uple list): [description]
    """
    
    distance_vector = []
    for _,classed in dataset.iterrows():
        tup = (dist_iris(flower, classed), classed)
        distance_vector.append(tup)
    # distance_vector = [ for classed in dataset ]
    ranked_distance = sorted(distance_vector, key=lambda t:t[0], reverse=True)
    knn_class = {}
    for rank in range(k):
        flow_class = ranked_distance[rank][1]['class']
        if not flow_class in knn_class:
            knn_class[flow_class] = 0
        knn_class[flow_class] +=1
    knn_found_class = max(knn_class, key=knn_class.get)
    print(knn_found_class, knn_class)
    return knn_found_class

def iris_classify(dataset):
    training_dataset_ratio = 0.66
    training_mask = np.random.rand(len(dataset)) < training_dataset_ratio
    test_mask = np.logical_not(training_mask)
    training_dataset = dataset[training_mask]

    test_dataset = dataset[test_mask]
    # print(test_dataset, training_dataset)
    print('training: {}\ntest: {}'.format(len(training_dataset), len(test_dataset)))
    correct_prediction = 0
    for index, tested in test_dataset.iterrows():
        # print(tested)
        knn_class = knn_iris(tested, training_dataset, 3)
        if knn_class  == tested['class']:
            correct_prediction += 1
        # break
    
    print('{} correct predictions'.format(correct_prediction))

def iris_vis(dataset):
    N = len(dataset)
    # presentation
    # plt.figure()
    sns.pairplot(dataset, hue='class')
    # distance matrix
    distance_matrix = np.zeros((N,N))
    byclass = dataset.sort_values(by=['class'])
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = dist_iris(byclass.iloc[i], byclass.iloc[j])
    plt.figure()
    sns.heatmap(distance_matrix)
    plt.show()



def iris():
    data = pandas.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    # print(data.head())

    
    # iris_classify(data)
    iris_vis(data)

    




iris()


# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors(X)
# print(indices)                                           
# # array([[0, 1],
# #        [1, 0],
# #        [2, 1],
# #        [3, 4],
# #        [4, 3],
# #        [5, 4]]...)
# print(distances)
# # array([[ 0.        ,  1.        ],
# #        [ 0.        ,  1.        ],
# #        [ 0.        ,  1.41421356],
# #        [ 0.        ,  1.        ],
# #        [ 0.        ,  1.        ],
# #        [ 0.        ,  1.41421356]])
