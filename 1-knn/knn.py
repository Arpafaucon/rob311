#/usr/bin/python3
# coding: utf-8
"""
K-nearest neighbor algorithm

Expects in the same directory the files:
- haberman.data
- haberman.names
- iris.data
- iris.names
"""

import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def euc_distance_function(num_features):
    """
    euclidean distance of a feature vector generator
    
    Args:
        num_features (int): number of features
    
    Returns:
        function (a, b) -> dist(a,b)
    """
    def dist(a, b):
        """
        distance between two array-like elements
        
        Returns:
            float: distance, positive real number
        """
        return sum( (a.iloc[i] - b.iloc[i])**2 for i in range(num_features) )
    return dist

def knn_classify(elt, training_dataset, distance_fn, K=3):
    """
    classify *elt* according to the data of *training_dataset*
    
    Args:
        elt (element descriptor): array of element features to be classified
        training_dataset (array[elt]): array of classified elements (elements must have the 'class' attribute)
        distance_fn (elt*elt->float): function giving the distance between two elements
        K (int, optional): Defaults to 3. number of nearest neighbours to work with
    
    Returns:
        elt_class: class of *elt*
    """
    # compute distance to all training element
    distance_to_elt = lambda x: distance_fn(x, elt)
    distance_array = [distance_to_elt(x) for _,x in training_dataset.iterrows()]
    # extract K minimal distances
    #   could be done manually by sorting and taking first K elts O(n log n)
    #   or by introselection: O(n)
    k_min_indices = np.argpartition(distance_array, K)[0:K]
    k_classes = {}
    # print(k_min_indices)
    # find the most represented class
    for ind in k_min_indices:
        train_class = training_dataset.iloc[ind]['class']
        if not train_class in k_classes:
            k_classes[train_class] = 0
        k_classes[train_class] += 1
    best_class = max(k_classes, key=k_classes.get)
    return best_class


def score_knn(test_dataset, training_dataset, distance_fn, K):
    """
    Score K-NN classification by the test_dataset
    
    Args:
        test_dataset (pandas.dataframe): test dataset
        training_dataset (pandas.dataframe): training dataset
        distance_fn (elt*elt->dist): distance function in the feature space
        K (int): number of nearest neighbours
    """
    correct_prediction = 0
    classes_list = list(training_dataset['class'].unique())
    class_indices = {v:i for i,v in enumerate(classes_list)}
    confusion_matrix = [[0]* len(classes_list) for _ in classes_list] 
    for _, tested in test_dataset.iterrows():
        # print(tested)
        
        knn_class = knn_classify(tested, training_dataset, distance_fn, K)
        
        if knn_class  == tested['class']:
            correct_prediction += 1
        knn_index = class_indices[knn_class]
        real_index = class_indices[tested['class']]
        confusion_matrix[real_index][knn_index] += 1
        # break
    print('correct: {:.2f} ( {} / {})'.format(correct_prediction/len(test_dataset), correct_prediction, len(test_dataset))) 
    print('confusion matrix:', confusion_matrix)
    return(correct_prediction, len(test_dataset), confusion_matrix, classes_list)


def split_dataset(dataset, training_ratio = 0.66):
    """
    Split the dataset into training and testing part
    The selection is random
    
    Args:
        dataset (dataframe): /
        training_ratio (float, optional): Defaults to 0.66. Ratio of lines that go to the training dataset
    
    Returns:
        (training, test): couple of datasets
    """
    training_mask = np.random.rand(len(dataset)) < training_ratio
    test_mask = np.logical_not(training_mask)
    return (dataset[training_mask], dataset[test_mask])

def iris(graphs = True):
    # iris dataset
    print('With IRIS dataset')
    data =  pandas.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    # if graphs:
    #     sns.pairplot(data, hue='class')
    #     plt.show()
    training, test = split_dataset(data)
    iris_distance = euc_distance_function(num_features=4)
    _, _, conf_mat, clist = score_knn(test, training, iris_distance, 3)
    if graphs:
        ax = sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=clist, yticklabels=clist)
        ax.set_xlabel('K-NN class')
        ax.set_ylabel('Real class')
        plt.show()
    

def hab(graphs=True):
    # haberman dataset
    print('With HABERMAN dataset')
    data =  pandas.read_csv('haberman.data', names=['age', 'year', 'node', 'class'])
    if graphs:
        sns.pairplot(data, hue='class', vars=['age', 'year', 'node'])
        plt.show()
    training, test = split_dataset(data)
    hab_distance = euc_distance_function(num_features=3)
    _, _, conf_mat, clist = score_knn(test, training, hab_distance, 3)
    if graphs:
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=clist, yticklabels=clist)
        plt.show()



def main():
    # select either dataset
    iris(graphs=not False)
    # hab()

if __name__ == '__main__':
    main()

# def dist_iris(a, b):
#     """
#     defines distances in the iris dataset
    
#     Args:
#         a (flower 4-uple): element
#         b (flower 4-uple): element

#     """
#     dist = [a.iloc[i] - b.iloc[i] for i in range(4)]
#     dist2 = [d**2 for d in dist]
#     res = sum(dist2)
#     return res

# def knn_iris(flower, dataset, k):
#     """
#     knn search for the flower
#     computes distance to all elements of the dataset and finds the winning class among the k nearest 
    
#     Args:
#         flower (flower 4-uple): 
#         dataset (classified flower 5-uple list): [description]
#     """
    
#     distance_vector = []
#     for _,classed in dataset.iterrows():
#         tup = (dist_iris(flower, classed), classed)
#         distance_vector.append(tup)
#     # distance_vector = [ for classed in dataset ]
#     ranked_distance = sorted(distance_vector, key=lambda t:t[0])
#     knn_class = {}
#     for rank in range(k):
#         flow_class = ranked_distance[rank][1]['class']
#         if not flow_class in knn_class:
#             knn_class[flow_class] = 0
#         knn_class[flow_class] +=1
#     knn_found_class = max(knn_class, key=knn_class.get)
#     print(knn_found_class, knn_class)
#     return knn_found_class
# def iris_classify(dataset):
#     training_dataset_ratio = 0.66
#     test_mask = np.logical_not(training_mask)
#     training_dataset = dataset[training_mask]

#     test_dataset = dataset[test_mask]
#     # print(test_dataset, training_dataset)
#     print('training: {}\ntest: {}'.format(len(training_dataset), len(test_dataset)))
#     correct_prediction = 0
#     for index, tested in test_dataset.iterrows():
#         # print(tested)
#         knn_class = knn_iris(tested, training_dataset, 3)
#         if knn_class  == tested['class']:
#             correct_prediction += 1
#         # break
    
#     print('{} correct predictions'.format(correct_prediction))

# def iris_vis(dataset):
#     N = len(dataset)
#     # presentation
#     # plt.figure()
#     sns.pairplot(dataset, hue='class')
#     # distance matrix
#     distance_matrix = np.zeros((N,N))
#     byclass = dataset.sort_values(by=['class'])
#     for i in range(N):
#         for j in range(N):
#             distance_matrix[i, j] = dist_iris(byclass.iloc[i], byclass.iloc[j])
#     plt.figure()
#     sns.heatmap(distance_matrix, cmap='magma'p).set_title('Distance matrix of classified data  ')
#     plt.show()



# def hab():
#     data = pandas.read_csv('haberman.data', names=['age','year','node','class'])

    





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
