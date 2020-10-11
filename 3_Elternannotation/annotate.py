from sys import argv


class Node(object):
    def __init__(self, tree, startpos, parent):
        self.children = []
        self.parent = parent
        if startpos < len(tree):
            if tree[startpos] != "(":  # case of reading a terminal symbol
                self.label, self.endpos = self.read_label(tree, startpos)
                if tree[self.endpos] != ")":  # for terminal symbol, endpos must be ")", otherwise it is an error
                    raise SyntaxError("error at -> " + tree[startpos:])

            else:  # case of reading a node
                self.label, self.endpos = self.read_label(tree, startpos + 1)
                while self.endpos < len(tree) and tree[self.endpos] != ")":
                    child = Node(tree, self.endpos, self)
                    if tree[self.endpos] == "(":  # the child is a non-terminal symbol
                        if tree[child.endpos + 1] == ")":  # if parent has no further child
                            self.endpos = child.endpos + 1  # set endpos to be ")" to stop the while-loop
                        elif tree[child.endpos + 1] == " ":  # if parent has another child
                            self.endpos = child.endpos + 2  # set endpos to "(" at front of the next child
                        else:
                            raise SyntaxError("error at -> " + tree[child.endpos:])
                    else:  # the child is a terminal symbol
                        self.endpos = child.endpos
                    self.children.append(child)

    def __str__(self):
        root = self.label
        parent_label = "None" if not self.parent else self.parent.label
        if len(self.children) > 0:
            root = "(" + self.label + "-" + parent_label
            for c in self.children:
                root += " "
                root += c.__str__()
        if len(self.children) > 0:
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
            if tree[pos].isalnum():
                label += tree[pos]
                pos += 1
            elif len(label) > 0:
                if tree[pos] == " ":
                    return label, pos + 1
                elif tree[pos] == ")":
                    return label, pos
                else:  # a label can either ends with a space or ")", otherwise it is an error
                    raise SyntaxError("error at -> " + tree[pos:])
            else:  # a label must have at least one character, otherwise it is an error
                raise SyntaxError("error at -> " + tree[pos:])


if __name__ == '__main__':
    with open(argv[1]) as file:
        for line in file:
            if not line.strip():  # skip empty line
                continue
            else:
                try:
                    root = Node(line, 0, None)
                    print(root)
                except SyntaxError as e:  # skip error line
                    print(e)
                    continue
