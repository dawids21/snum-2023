import math

import pandas as pd
import pydot

# Step 1: Load and Preprocess the Data
data = pd.read_csv("titanic-homework.csv")
data = data.drop(columns=["PassengerId", "Name"])

data["Age"] = pd.cut(
    data["Age"],
    bins=[0, 20, 40, 100],
    labels=["young", "middle", "old"],
    right=False,
)


# Step 2: Define Helper Functions
def calculate_entropy(data):
    classes = data["Survived"].unique()
    entropy = 0
    total_samples = len(data)

    for c in classes:
        p = len(data[data["Survived"] == c]) / total_samples
        entropy += -p * math.log2(p)

    return entropy


def calculate_conditional_entropy(data, attribute):
    conditional_entropy = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        p_value = len(subset) / len(data)
        conditional_entropy += p_value * calculate_entropy(subset)

    return conditional_entropy


def calculate_information_gain(data, attribute):
    return calculate_entropy(data) - calculate_conditional_entropy(data, attribute)


def calculate_gain_ratio(data, attribute):
    info_gain = calculate_information_gain(data, attribute)
    split_info = -sum(
        (len(data[data[attribute] == value]) / len(data))
        * math.log2(len(data[data[attribute] == value]) / len(data))
        for value in data[attribute].unique()
    )

    return info_gain / split_info


# Step 3: Build the Decision Tree
def build_decision_tree(data, depth=0):
    if len(data["Survived"].unique()) == 1:
        return {}

    if depth >= 5:
        return {}

    best_attribute = max(data.columns[:-1], key=lambda a: calculate_gain_ratio(data, a))
    tree = {best_attribute: {}}

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        subset = subset.drop(columns=[best_attribute])
        subtree = build_decision_tree(subset, depth + 1)
        subtree["survived"] = len(subset[subset["Survived"] == 1])
        subtree["not survived"] = len(subset[subset["Survived"] == 0])
        tree[best_attribute][str(value)] = subtree

    return tree


def draw(tree, parent=None):
    if len(tree.keys()) == 2:
        return
    node_name = list(tree.keys())[0]
    if parent is not None:
        node = pydot.Node(name=parent.get_name() + node_name, label=node_name, shape="box")
    else:
        node = pydot.Node(name=node_name, label=node_name, shape="box")
    graph.add_node(node)
    if parent is not None:
        graph.add_edge(pydot.Edge(parent, node))
    for k, v in tree[node_name].items():
        k_node = pydot.Node(name=node.get_name() + k,
                            label=k + "\nSurvived: " + str(v["survived"]) + "\nNot survived: " + str(v["not survived"]),
                            shape="box")
        graph.add_node(k_node)
        graph.add_edge(pydot.Edge(node, k_node))
        draw(v, parent=k_node)


decision_tree = build_decision_tree(data)

graph = pydot.Dot(graph_type='graph')
draw(decision_tree)
graph.write_png('example1_graph.png')
