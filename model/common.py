import numpy as np
import pandas as pd
from sklearn import preprocessing
from treelib import Tree


def create_hiarchy_tree(
    categories_path="data/category_parent.csv", clothing_categories=None
):
    """
    construct categories hieararchy tree from csv file & dict of ids and categories names
    """
    categories = pd.read_csv(categories_path)
    categories = categories.sort_values(by="category_id")
    categories = categories.fillna(-1)
    categories = categories.astype(int)
    tree = Tree()
    tree.create_node(-1, -1)  # root node
    if clothing_categories is None:
        for i, row in categories.iterrows():
            if i != 0:
                tree.create_node(
                    row["category_id"], row["category_id"], parent=row["parent_id"]
                )
        return tree
    else:
        categories["label"] = categories["category_id"].map(clothing_categories)
        for i, row in categories.iterrows():
            if i != 0:
                tree.create_node(
                    row["label"], row["category_id"], parent=row["parent_id"]
                )
        return tree


def collapse_children(tree, id, level):
    # collapse categories
    # eg : collapse_children(tree, 77, 4) returns  -1

    while level != 0:
        level -= 1
        id = tree.parent(id).identifier
    return id


def collapse_encoded_targets(label_enconders, tree, array, start_level, end_level):
    # array is encoded flat preds

    # reverse encode
    decoded_array = label_enconders[start_level].inverse_transform(array)

    # collapse
    collapsed_decoded = np.array(
        [collapse_children(tree, i, end_level) for i in decoded_array]
    )

    # encode for specific level
    return label_enconders[end_level].transform(collapsed_decoded)
