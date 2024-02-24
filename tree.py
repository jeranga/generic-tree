import pandas as pd
import numpy as np
from math import log2
import random
import time

class DataPoint:
    """
    Represents a single data point in the dataset.
    """
    def __init__(self, label, features):
        """
        Initializes a DataPoint object with a label and features.
        :param label: The label or target value of the data point.
        :param features: A list of features for the data point.
        """
        self.label = label
        self.features = features

    def __str__(self):
        """
        Returns a string representation of the data point.
        """
        return "< " + str(self.label) + ": " + str(self.features) + " >"

def get_data(filename):
    """
    Loads data from a CSV file and returns a list of DataPoint objects.
    :param filename: Path to the CSV file.
    :return: A list of DataPoint objects.
    """
    data = []
    df = pd.read_csv(filename, header=None)
    for _, row in df.iterrows():
        row_list = row.tolist()
        label = row_list[-1]
        features = row_list[:-1]
        data_point = DataPoint(label, features)
        data.append(data_point)
    return data

class TreeNode:
    """
    Represents a node in the decision tree.
    """
    def __init__(self):
        self.is_leaf = True
        self.feature_idx = None
        self.thresh_val = None
        self.prediction = None
        self.left_child = None
        self.right_child = None

    def printTree(self, level=0):
        """
        Prints the tree from this node downwards.
        :param level: Current level in the tree (used for indentation).
        """
        if self.is_leaf:
            print('-'*level + 'Leaf Node:      predicts ' + str(self.prediction))
        else:
            print('-'*level + 'Internal Node:  splits on feature ' 
                  + str(self.feature_idx) + ' with threshold ' + str(self.thresh_val))
            self.left_child.printTree(level+1)
            self.right_child.printTree(level+1)

def make_prediction(node, data):
    """
    Makes a prediction for a given data point using the decision tree.
    :param node: The current node in the decision tree.
    :param data: The DataPoint for which to make the prediction.
    :return: The prediction.
    """
    if node.is_leaf:
        return node.prediction
    
    currData = data.features[node.feature_idx]
    if currData < node.thresh_val:
        return make_prediction(node.left_child, data)
    else:
        return make_prediction(node.right_child, data)

def split_dataset(data, feature_idx, threshold):
    """
    Splits the dataset into two parts based on a threshold value of a feature.
    :param data: List of DataPoint objects to split.
    :param feature_idx: Index of the feature to use for splitting.
    :param threshold: Threshold value for the feature.
    :return: A tuple of two lists (left_split, right_split).
    """
    left_split, right_split = [], []
    for point in data:
        if point.features[feature_idx] < threshold:
            left_split.append(point)
        else:
            right_split.append(point)
    return (left_split, right_split)

def count_classifications(data):
    """
    Determines the majority classification in a dataset.
    :param data: List of DataPoint objects.
    :return: The majority classification.
    """
    count = sum(entry.label for entry in data)
    total = len(data)
    return 1 if count >= (total / 2) else 0

def calc_entropy(data):
    """
    Calculates the entropy of a dataset.
    :param data: List of DataPoint objects.
    :return: The calculated entropy.
    """
    count = sum(entry.label for entry in data)
    return entropy(count, len(data) - count)

def entropy(left_count, right_count):
    """
    Calculates the entropy given counts of two classes.
    :param left_count: Count of the first class.
    :param right_count: Count of the second class.
    :return: The entropy value.
    """
    total = left_count + right_count
    if left_count == 0 or right_count == 0:
        return 0
    return (-left_count/total) * log2(left_count/total) - (right_count/total) * log2(right_count/total)

def entropy_tot(left, right):
    """
    Calculates the total entropy for a split.
    :param left: The left subset after the split.
    :param right: The right subset after the split.
    :return: The total entropy.
    """
    total = len(left) + len(right)
    ent_left = calc_entropy(left)
    ent_right = calc_entropy(right)
    split_ent = (ent_left * (len(left) / total)) + (ent_right * (len(right) / total))
    return split_ent

def calc_best_threshold(data, feature_idx):
    """
    Calculates the best threshold for splitting the dataset on a given feature.
    :param data: List of DataPoint objects.
    :param feature_idx: Index of the feature to consider.
    :return: A tuple containing the best information gain and the corresponding threshold.
    """
    best_info_gain = 0.0
    best_thresh = None
    data_sorted = sorted(data, key=lambda x: x.features[feature_idx])
    if data_sorted[0].features[feature_idx] == data_sorted[-1].features[feature_idx]:
        return (best_info_gain, best_thresh)
    
    starting_ent = calc_entropy(data_sorted)
    
    for idx in range(len(data_sorted) - 1):
        val = data_sorted[idx].features[feature_idx]
        next_val = data_sorted[idx + 1].features[feature_idx]
        if val != next_val:
            curr_thresh = (val + next_val) / 2
            left, right = split_dataset(data_sorted, feature_idx, curr_thresh)
            curr_ent = entropy_tot(left, right)
            curr_gain = starting_ent - curr_ent
            if curr_gain > best_info_gain:
                best_info_gain = curr_gain
                best_thresh = curr_thresh
    return (best_info_gain, best_thresh)

def identify_best_split(data):
    """
    Identifies the best feature and threshold for splitting the dataset.
    :param data: List of DataPoint objects.
    :return: A tuple containing the index of the best feature and the best threshold.
    """
    best_feature = None
    best_thresh = None
    best_info_gain = 0.0
    for i in range(len(data[0].features)):
        curr_IG, curr_thresh = calc_best_threshold(data, i)
        if curr_IG > best_info_gain:
            best_info_gain = curr_IG
            best_feature = i
            best_thresh = curr_thresh
    return (best_feature, best_thresh)

def create_leaf_node(data):
    """
    Creates a leaf node for the decision tree.
    :param data: The subset of the dataset corresponding to this leaf.
    :return: A TreeNode configured as a leaf node.
    """
    node = TreeNode()
    node.prediction = count_classifications(data)
    return node

def create_decision_tree(data, max_levels):
    """
    Recursively builds a decision tree.
    :param data: The dataset to build the tree from.
    :param max_levels: The maximum depth of the tree.
    :return: The root node of the decision tree.
    """
    if calc_entropy(data) == 0 or max_levels == 1:
        return create_leaf_node(data)
    
    currNode = TreeNode()
    currNode.is_leaf = False
    currNode.feature_idx, currNode.thresh_val = identify_best_split(data)
    if currNode.thresh_val is None:
        return create_leaf_node(data)
    left, right = split_dataset(data, currNode.feature_idx, currNode.thresh_val)
    if len(left) > 0:
        currNode.left_child = create_decision_tree(left, max_levels-1)
    if len(right) > 0:
        currNode.right_child = create_decision_tree(right, max_levels-1)

    return currNode

def calc_accuracy(tree_root, data):
    """
    Calculates the accuracy of the decision tree on a given dataset.
    :param tree_root: The root node of the decision tree.
    :param data: The dataset to test.
    :return: The accuracy as a percentage.
    """
    correct_count = sum(1 for point in data if make_prediction(tree_root, point) == point.label)
    total_count = len(data)
    return correct_count / total_count

def store_file_name():
    """
    Prompts the user to enter a file name and validates its existence.
    :return: The validated file name.
    """
    while True:
        file_name = input("Please enter the file name (csv) you wish to build the model for: ")
        try:
            with open(file_name, 'r') as file:
                print("Building Decision Tree...")
                break
        except FileNotFoundError:
            print("File not found. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.") 
    return file_name

def k_fold_cross_validation(data, k=5, max_depth=10):
    """
    Performs K-fold cross-validation on the dataset and prints the average accuracy.
    :param data: The dataset to be used in cross-validation.
    :param k: The number of folds.
    :param max_depth: The maximum depth of the decision tree.
    """
    n = len(data)
    fold_size = int(n / k)
    total_accuracy = 0.0

    for i in range(k):
        print(f"Running fold {i+1}/{k}...")
        test_index_start = fold_size * i
        test_index_end = fold_size * (i + 1)
        
        test_set = data[test_index_start:test_index_end]
        train_set = data[:test_index_start] + data[test_index_end:]

        start_time = time.time()
        tree = create_decision_tree(train_set, max_depth)
        end_time = time.time()

        accuracy = calc_accuracy(tree, test_set)
        total_accuracy += accuracy

        print(f"Fold {i+1} Accuracy: {accuracy*100:.2f}%")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")

    average_accuracy = (total_accuracy / k) * 100
    print(f'The average accuracy across all folds is: {average_accuracy:.2f}%')

def main():
    """
    The main execution flow of the script.
    """
    file_name = store_file_name()
    
    print("Loading data...")
    data = get_data(file_name)
    random.shuffle(data)

    print("Starting K-Fold Cross Validation...")
    k_fold_cross_validation(data, k=5, max_depth=10)

if __name__ == "__main__":
    main()