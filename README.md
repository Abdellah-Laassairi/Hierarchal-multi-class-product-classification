# Hierarchal multi-class usecase : Product Classification
On an e-commerce website, it is important to place the right products in the correct categories. When this is not the case, they may go unnoticed by buyers and can also lead to a poor user experience, especially if they are present in large quantities in the wrong category.

Each e-commerce website has its own taxonomy of categories. Here is a dummy subtree example for our dataset:
![assets](assets/subtree.png)

A product can be described by various and diverse pieces of information. In the context of this study, the explanatory variables have already been transformed and reduced to a dimension of 128.

The objective of this study is to create a classification algorithm for the leaf categories (category_id) based on these 128 features (f0, ... f127).

To install the necessary packages and libraries :

```
pip install -r requirements.txt
```

To launch training using a BigBang approach :

```
python3 -m train fit -c config.yaml --exp-path runs/big-bang
```

You can then use tensorboard to visualize all the different metrics :

```
tensorboard --logdir=runs
```

## Provided Data:

- data_train.csv: It contains the following columns:
  - product_id: Product IDs
  - category_id: Leaf categories of the products
  f0 ... f127: The 128 features
- data_test.csv: Same format as data_train.csv
- category_parent.csv: Contains pairs of IDs, each corresponding to a category and its parent, with -1 representing the root category.

Note: As some transformations have already been applied to the data, the variables are not interpretable.

## Objectives & Guidelines:

- Analyze the data and briefly describe some possible classification algorithms for such a dataset.
- Choose and implement at least one of these algorithms using the training set (data_train.csv), make predictions on the provided test set (data_test.csv). Specify the chosen metrics and explain your approach.
- Perform an analysis of the obtained results.
- Do not use the category hierarchy as an explanatory variable.
