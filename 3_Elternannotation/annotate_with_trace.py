from sys import argv


class Node(object):
    trace_dict = {}

    def __init__(self, tree, startpos, parent, root):
        """read a trie that is marked with trace features."""

        self.children = []
        self.parent = parent
        self.filler = None
        self.trace = None
        self.trace_list = []
        self.root = self if root is None else root
        if startpos < len(tree):
            if tree[startpos] != "(":
                self.label, self.endpos = self.read_label(tree, startpos)
                self.read_trace_label(self)
                if tree[self.endpos] != ")":
                    raise SyntaxError("error at -> " + tree[startpos:])

            else:
                self.label, self.endpos = self.read_label(tree, startpos + 1)
                self.read_trace_label(self)
                while self.endpos < len(tree) and tree[self.endpos] != ")":
                    child = Node(tree, self.endpos, self, self.root)
                    if tree[self.endpos] == "(":
                        if tree[child.endpos + 1] == ")":
                            self.endpos = child.endpos + 1
                        elif tree[child.endpos + 1] == " ":
                            self.endpos = child.endpos + 2
                        else:
                            raise SyntaxError("error at -> " + tree[child.endpos:])
                    else:
                        self.endpos = child.endpos
                    self.children.append(child)

    def read_trace_label(self, node):
        if not node.label.isalnum():  # read trace
            if node.label[:3] == "*T*" and node.label[-1].isdigit():
                node.trace = node.label[-1]
                node.label = "*T*"
            elif node.label[-1].isdigit():  # read filler
                node.filler = node.label[-1]
                node.label = node.label[:-2]
                self.trace_dict[node.filler] = node.label
            else:
                raise SyntaxError("trace or filler is wrongly written")

    def add_trace_features(self, node):
        """add traces features to the trie."""
        if node.children:
            if node.filler:
                index = node.filler
                self.annotate_trace_up(node.root, index)
            for c in node.children:
                self.add_trace_features(c)

    def annotate_trace_up(self, root, index):
        """
        search from the root node for the trace of the given index
        and annotate parent nodes with the trace bottom up.
        """
        if not root.children:
            if root.trace == index:
                self.annotate_trace_down(root, index)
        else:
            for c in root.children:
                self.annotate_trace_up(c, index)

    def annotate_trace_down(self, node, index):
        """
        annotate parent nodes of a given node with the trace of a given index, except the root node.
        """
        while node.parent and node.parent.root != node.parent:
            node.parent.trace_list.append(index)
            node = node.parent

    def __str__(self):
        if not self.children:
            root = self.label
        else:
            parent_label = "Root" if self.parent is None else self.parent.label
            root = "(" + self.label + "-" + parent_label
            if self.filler:
                root += "\\" + self.trace_dict[self.filler]
            if self.trace_list:
                for i in self.trace_list:
                    root += "/" + self.trace_dict[i]
            for c in self.children:
                root += " "
                root += c.__str__()
            root += ")"

        return root

    def read_label(self, tree, pos):
        """
        Read a label of a given trie from a given position. Return the label and the end position.

        In case of a terminal symbol, end position is set to be at the next ")".
        In case of a non-terminal symbol, end position is set to be at either "(" or the next terminal label.

        """
        label = ""
        while pos < len(tree):
            if tree[pos] not in ["(", ")", " "]:
                label += tree[pos]
                pos += 1
            elif len(label) > 0:
                if tree[pos] == " ":
                    return label, pos + 1
                elif tree[pos] == ")":
                    return label, pos
                else:
                    raise SyntaxError("error at -> " + tree[pos:])
            else:
                raise SyntaxError("error at -> " + tree[pos:])


if __name__ == '__main__':
    with open(argv[1]) as file:
        for line in file:
            if not line.strip():
                continue
            else:
                try:
                    root = Node(line, 0, None, None)
                    root.add_trace_features(root)
                    print(root)
                except SyntaxError as e:
                    print(e)
                    continue
