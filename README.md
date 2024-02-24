# Decision Tree Classifier: From Scratch

Welcome my Decision Tree Classifier repository! This project is a pure Python implementation of a decision tree classifier, designed to demystify the mechanics of decision trees and provide a hands-on learning experience. By building this classifier from the ground up, I aim to offer insights into the data preprocessing, recursive tree building, node splitting based on entropy, and the evaluation process using K-fold cross-validation. 

## Features

- **Data Loading**: Utilizes Pandas for efficient data loading and preprocessing, ensuring compatibility with various CSV datasets.

- **Custom Data Structure**: Implements a `DataPoint` class to encapsulate features and labels, facilitating easy data manipulation.

- **Recursive Tree Building**: Employs a recursive approach to grow the decision tree from the root, creating a clear and logical tree structure.

- **Entropy-Based Splitting**: Utilizes entropy and information gain to determine the optimal feature and threshold for splitting, ensuring the most informative decisions are made at each node.

- **K-Fold Cross-Validation**: Implements K-fold cross-validation from scratch, providing a robust method to evaluate the classifier's performance across different data subsets.

- **TreeNode Class**: A versatile class that captures both leaf and internal nodes, including properties for split criteria, predictions, and child nodes, offering a comprehensive view of the decision tree.

- **Modular Design**: The code is organized into modular functions with detailed comments and docstrings, making it easy to understand, modify, and extend.

- **Configurable Parameters**: Allows users to easily adjust the depth of the tree and the number of folds in cross-validation, making the classifier adaptable to a wide range of datasets.

## Getting Started

To use this decision tree classifier, clone this repository and ensure you have Python 3.x installed along with Pandas and NumPy libraries.

1. Clone the repository:
```bash
git clone <repository-url>
```
2. Install dependencies:
```bash
pip install pandas numpy
```
3. Run the classifier with your dataset:
```bash
python decision_tree.py
```
Replace decision_tree.py with the path to the script if you're running it from a different directory.

## Dataset Specifications
To ensure optimal performance with this classifier, datasets should adhere to the following format:

- **Feature Columns:** All columns except the last should represent features. Features should be numerical or one-hot encoded if categorical.

- **Label Column:** The final column must be the label or target variable.

- **Preprocessing:** Ensure there are no missing values. If categorical features are present, they should be one-hot encoded prior to running the classifier.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!
