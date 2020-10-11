from sys import argv

ID = 0


class Node(object):
    def __init__(self, char, node_id):
        self.char = char
        self.node_id = node_id
        self.target_node = {}  # contains child nodes
        self.final = False  # True if at least a word ends at this node, otherwise False
        self.start_node = -1  # the start node of a root node is defined as -1
        global ID
        ID += 1

    def __repr__(self):
        """
        Root node is represented as "-1    root    0".
        A normal node is represented as "0    i    1".
        A node marked as word ending is represented as "0    i_    1".
        Note: here an underscore is appended to the character instead of a space,
        so that it is easier to check if the output is correct.
        """
        # a root node is initialized with Node(-1, None, 0)
        char = "root" if not self.char else str(self.char)
        if self.final:
            char += "_"
        start = str(self.start_node)
        return "%s\t%s\t%s" % (start, char, self.node_id)

    def add_word_to_tree(self, word):
        """Read a word character-by-character and add it to the tree."""
        global ID
        if word:
            first_char = word[0]
            if first_char in self.target_node.keys():
                if len(word) == 1:
                    self.target_node[first_char].final = True
                else:
                    self.target_node[first_char].add_word_to_tree(word[1:])

            else:
                self.create_child_node(word, ID)

    def create_child_node(self, word, node_id):
        """Create new nodes from each character in the given word and add them to the tree."""
        new_child = Node(word[0], node_id)
        self.target_node[word[0]] = new_child
        new_child.start_node = self.node_id
        if len(word) == 1:
            new_child.final = True
        else:
            global ID
            new_child.create_child_node(word[1:], ID)

    def add_child(self, node):
        """
        Add the given node to target_node.
        This method is used in the lookup process.
        """
        if node.char not in self.target_node.keys():
            self.target_node[node.char] = node
            node.start_node = self.node_id
        else:  # if the tree is correct, there should not be any duplicate character in target_node
            raise Exception

    def print_tree(self):
        print(self)
        for _, c in self.target_node.items():
            c.print_tree()


if __name__ == "__main__":
    root = Node(None, ID)
    word_list = argv[1]
    with open(word_list, "r") as f:
        for line in f:
            root.add_word_to_tree(line.strip())
        root.print_tree()
