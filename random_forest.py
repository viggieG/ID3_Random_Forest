import random, parse, node
from node import Node
# from parse import parse
from ID3_old import ID3, classify, prune, most_common_class, test
from collections import Counter


def random_forest(filename, num_trees=None, max_features=None, max_depth=None):

    data_list = parse.parse(filename)

    attributes = data_list[0].keys()



    # Split data into training, validation and testing sets (50-25-25 split for now)
    random.shuffle(data_list)

    training_data = data_list[:len(data_list) // 2]
    valid_data = data_list[len(data_list) // 2:3 * len(data_list) // 4]
    test_data = data_list[3 * len(data_list) // 4:]



    # Simple oversampling to handle class imbalance
    class_counts = Counter([data["Class"] for data in training_data])
    max_count = max(class_counts.values())

    for class_val, count in class_counts.items():
        if count < max_count:
            extra_samples = [data for data in training_data if data["Class"] == class_val]
            training_data.extend(extra_samples * (max_count - count))

    trees = []
    for _ in range(num_trees):
        sampled_data = random.choices(training_data, k=len(training_data))
        if max_features is None:
            max_features = len(sampled_data[0]) - 1  # minus one for the "Class" key
        features = random.sample(list(sampled_data[0].keys())[:-1], max_features)  # excluding "Class"
        decision_tree = ID3_wrapper(sampled_data, max_features=max_features, max_depth=max_depth)

        prune(decision_tree, sampled_data)
        trees.append(decision_tree)

    def random_forest_predict(trees, example):
        predictions = [classify(tree, example) for tree in trees]
        return most_common_class(predictions)

    correct_predictions = sum(1 for data in test_data if random_forest_predict(trees, data) == data["Class"])
    accuracy_rf = correct_predictions / len(test_data)

    return accuracy_rf




def ID3_wrapper(examples, max_features=None, max_depth=None):
    if max_depth is not None and max_depth <= 0:
        node = Node
        node.label = most_common_class(examples)
        return node

    # If max_features is specified, sample the attributes
    if max_features:
        all_attributes = list(examples[0].keys())
        all_attributes.remove("Class")
        sampled_attributes = random.sample(all_attributes, max_features)

        # Filter examples to only include sampled attributes and "Class"
        filtered_examples = [{attr: ex[attr] for attr in sampled_attributes + ["Class"]} for ex in examples]
    else:
        filtered_examples = examples

    # Call the original ID3 function
    tree = ID3(filtered_examples)

    # If max_depth is specified, adjust the tree depth
    if max_depth is not None:
        # Logic to truncate the tree to respect max_depth can be added here.
        # Note: Implementing this might be a bit involved.
        pass

    return tree


if __name__ == "__main__":
    filename = "candy.data"
    rf_accuracy = random_forest(filename,num_trees=100, max_features=7, max_depth=8)  # You can tweak max_features and max_depth
    print(f"Random Forest Accuracy on test_data: {rf_accuracy:.2f}")
