from sys import argv
from importlib import import_module
build_tree = import_module("build-tree")  # module name containing dash(-) can not be imported directly


def generate_node(start_id, node_char, node_id):
    """generate a node from the given arguments"""
    if start_id == -1 and node_char == "root":  # identify root node
        node = build_tree.Node(node_char, node_id)
    elif len(node_char) > 1 and node_char[1] == "_":  # check if the node has the final marker
        node = build_tree.Node(node_char[0], node_id)
        node.final = True
    elif len(node_char) == 1:
        node = build_tree.Node(node_char, node_id)
    else:
        raise Exception("Tree file is not correct")
    node.start_node = start_id
    return node


def lookup(tree, word):
    """return True if word is in tree, return False otherwise"""
    result = False
    if word:
        first_char = word[0]
        if first_char in tree.target_node.keys():  # in case the current character is in target_node
            if len(word) == 1 and tree.target_node[first_char].final:  # check if the last character is reached and the node is marked as final
                result = True
            else:  # if it is not the last character, then check the next character further down the tree
                result = lookup(tree.target_node[first_char], word[1:])
    return result


def read_tree(tree_file):
    """Read a stored tree from file into Node object. Return the root node."""
    node_dict = {}  # a key is the node id, a value is a Node object
    with open(tree_file, "r") as t:
        for line in t:
            if line:
                line = line.strip()
                start_id, node_char, node_id = line.split("\t")
                start_id = int(start_id)
                node_id = int(node_id)

                node = generate_node(start_id, node_char, node_id)
                if start_id == -1:  # set up root node
                    node_dict[0] = node
                elif start_id not in node_dict.keys():
                    node_dict[start_id] = node
                else:
                    node_dict[start_id].add_child(node)
                    node_dict[node_id] = node
        root = node_dict[0]
    return root

if __name__ == "__main__":
    tree = argv[1]
    words = argv[2]
    root = read_tree(tree)

    with open(words, "r") as f:
        for line in f:
            line = line.strip()
            result = lookup(root, line)
            result_print = "known" if result else "unknown"
            print(line, result_print, sep="\t")
