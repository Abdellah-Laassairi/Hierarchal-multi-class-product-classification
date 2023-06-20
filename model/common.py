import numpy as np
import pandas as pd
from sklearn import preprocessing
from treelib import Tree


def create_hiarchy_tree(
    categories_path="data/category_parent.csv", clothing_categories=None
):
    """
    Construct a categories hierarchy tree from a CSV file and a dictionary of ids and category names.

    Args:
        categories_path (str): Path to the CSV file containing category-parent information.
        clothing_categories (dict): Dictionary mapping category ids to category names.

    Returns:
        Tree: Constructed categories hierarchy tree.
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
    """
    Collapse categories in the hierarchy tree.

    Args:
        tree (Tree): Hierarchy tree containing category nodes.
        id: Identifier of the starting category.
        level (int): The level to which categories should be collapsed.

    Returns:
        int: Identifier of the collapsed category.
    """
    while level != 0:
        level -= 1
        id = tree.parent(id).identifier
    return id


def collapse_encoded_targets(label_enconders, tree, array, start_level, end_level):
    """
    Collapse encoded targets based on the specified start and end levels.

    Args:
        label_enconders (list): List of label encoders for different levels.
        tree (Tree): Hierarchy tree containing category nodes.
        array (ndarray): Encoded flat predictions.
        start_level (int): Start level for collapsing.
        end_level (int): End level for collapsing.

    Returns:
        ndarray: Encoded collapsed predictions for the specified end level.
    """
    # Reverse encode
    decoded_array = label_enconders[start_level].inverse_transform(array)

    # Collapse
    collapsed_decoded = np.array(
        [collapse_children(tree, i, end_level) for i in decoded_array]
    )

    # Encode for specific level
    return label_enconders[end_level].transform(collapsed_decoded)
