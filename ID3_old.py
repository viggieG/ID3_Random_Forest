from node import Node
import math
from collections import defaultdict

def information_gain(data, attribute, entropy_dataset):
    total_examples = len(data)
    attribute_values = defaultdict(list)
    missing_data = []  # data entries where attribute value is missing

    for entry in data:
        if entry[attribute] == '?':
            missing_data.append(entry)
        else:
            attribute_values[entry[attribute]].append(entry)

    # Calculate original weighted entropy
    weighted_entropy = 0
    for value_list in attribute_values.values():
        prob = len(value_list) / total_examples
        weighted_entropy += prob * entropy(value_list)

    # If there are missing values, distribute them among the existing values
    if missing_data:
        missing_weight = len(missing_data) / total_examples
        for value, value_list in attribute_values.items():
            extended_list = value_list + missing_data
            prob = len(extended_list) / total_examples
            weighted_entropy += missing_weight * prob * entropy(extended_list)

    return entropy_dataset - weighted_entropy


def ID3(examples, default=None):
    node = Node()
    if not examples:
        node.label = default
        return node

    class_values = [entry["Class"] for entry in examples]
    if len(set(class_values)) == 1:
        node.label = class_values[0]
        return node

    attributes = set(examples[0].keys()) - {"Class"}
    if not attributes:
        node.label = most_common_class(examples)
        return node

    entropy_dataset = entropy(examples)
    gains = {attribute: information_gain(examples, attribute, entropy_dataset) for attribute in attributes if attribute != "Class"}
    max_attribute = max(gains, key=gains.get)

    # Threshold for minimum information gain
    if gains[max_attribute] < 0.01: 
        node.label = most_common_class(examples)
        return node

    node.attribute = max_attribute
    for value in set(entry[max_attribute] for entry in examples if entry[max_attribute] != '?'):
        subset = [entry for entry in examples if entry[max_attribute] == value]
        node.children[value] = ID3(subset, default=most_common_class(examples))

    return node



def most_common_value(data, attribute):
    values = [entry[attribute] for entry in data if entry[attribute] != '?']
    return max(set(values), key=values.count)

def replace_unknown_values(data):
    for attribute in data[0]:
        if attribute != "Class":
            most_common = most_common_value(data, attribute)
            for entry in data:
                if entry[attribute] == '?':
                    entry[attribute] = most_common
    return data

def entropy(data):
    total_examples = len(data)
    label_count = defaultdict(int)
    
    for entry in data:
        label = entry["Class"]
        label_count[label] += 1
        
    entropy = 0
    for label in label_count:
        prob = label_count[label] / total_examples
        entropy -= prob * math.log2(prob)
    return entropy
 
def evaluate(node, example):
    if node.label is not None:
        return node.label
    elif node.attribute:
        attribute_value = example[node.attribute]
        if attribute_value in node.children:
            return evaluate(node.children[attribute_value], example)
        else:
            # Handle cases where the attribute value from the example isn't in the tree.
            return most_common_class_in_children(node) or most_common_class([example])
    else:
        # This is an unexpected state, a node without an attribute and a label.
        # For now, let's return the most common class in the data to handle this gracefully.
        return most_common_class([example])

 
def test(node, examples):
    correct_predictions = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            correct_predictions += 1
    return correct_predictions / len(examples)

def prune(node, examples):
    # Base case: if the node is a leaf, return
    if not node.children:
        return
    
    # First, let's recursively prune the children
    for value, child_node in node.children.items():
        subset = [entry for entry in examples if entry[node.attribute] == value]
        if subset:
            prune(child_node, subset)

    # Store the current state of the node
    original_children = node.children
    original_attribute = node.attribute

    # Prune the current node (i.e., make it a leaf)
    node.children = {}
    node.attribute = None
    node.label = most_common_class(examples)
    
    # Check if pruning improves accuracy
    pruned_accuracy = accuracy(node, examples)
    original_accuracy = accuracy_after_pruning(node, original_attribute, original_children, examples)
    
    # If the original accuracy is better, revert the changes
    if original_accuracy >= pruned_accuracy:
        node.children = original_children
        node.attribute = original_attribute
        node.label = None


def accuracy(node, examples):
    if not examples:
        return 0  # Return 0 accuracy if there are no examples
    
    correct_predictions = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            correct_predictions += 1
    
    return correct_predictions / len(examples)

def classify(node, example):
    while node.attribute:
        if example[node.attribute] in node.children:
            node = node.children[example[node.attribute]]
        else:
            # If the attribute value is not found in the current node's children
            return most_common_class_in_children(node) or most_common_class([example])
    return node.label if node.label is not None else most_common_class([example])


def most_common_class(examples):
    if not examples:
        return None
    # Check if the examples are dictionaries or direct class labels
    if isinstance(examples[0], dict):
        class_values = [entry["Class"] for entry in examples]
    else:
        class_values = examples
    return max(class_values, key=class_values.count)


def most_common_class_in_children(node):
    if not node.children:
        # Return a default value when there are no children nodes
        return None
    counts = defaultdict(int)
    for child in node.children.values():
        counts[child.label] += 1
    return max(counts, key=counts.get)


def accuracy_after_pruning(node, original_attribute, original_children, examples):
    current_label = node.label
    node.attribute = original_attribute
    node.children = original_children
    node.label = None
    
    current_accuracy = accuracy(node, examples)
    
    node.label = current_label
    return current_accuracy

def accuracy_without_pruning(node, examples):
    if not examples:
        return 0  # Return 0 accuracy if there are no examples
    
    correct_predictions = 0
    for example in examples:
        if evaluate(node, example) == example["Class"]:
            correct_predictions += 1
    
    return correct_predictions / len(examples)

