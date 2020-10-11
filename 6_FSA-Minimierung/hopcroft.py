from sys import argv

ID = 0


class Node(object):
    def __init__(self):
        global ID
        self.in_edges = []  # list of (incoming character, incoming node)
        self.out_edges = {}  # dict of outgoing character and target node
        self.node_id = ID
        self.visited = False
        self.is_fail_node = False
        ID += 1

    def __repr__(self):
        return "%s" % (self.node_id)

    def print_all(self, print_fail_node=True):
        self.print_node(print_fail_node)
        self.reset_visited()

    def print_node(self, print_fail_node=True):

        if not print_fail_node:
            if self.is_fail_node:
                self.visited = True
        if not self.visited:
            if self.node_id == 0:
                result = "%s\t%s\t%s" % (-1, "root", self.node_id)
                print(result)
            for char, in_node in self.in_edges:
                result = "%s\t%s\t%s" % (in_node.node_id, char, self.node_id)
                print(result)
            self.visited = True
            for _, out_node in self.out_edges.items():
                out_node.print_node(print_fail_node)

    def add_word(self, word):
        """add a word to this node"""
        if word:
            char = word[0]
            if char not in self.out_edges.keys():
                new_node = Node()
                self.out_edges[char] = new_node
                new_node.in_edges.append((char, self))
                if len(word) == 1:
                    final_node = Node()
                    new_node.out_edges[" "] = final_node
                    final_node.in_edges.append((" ", new_node))
                else:
                    new_node.add_word(word[1:])
            else:
                self.out_edges[char].add_word(word[1:])

    def reset_visited(self):
        """set self.visited of all nodes to be False"""
        self.visited = False
        if not self.is_fail_node:
            for _, target_node in self.out_edges.items():
                target_node.reset_visited()

    def trie_to_DEA(self):
        """convert the given trie into a DEA"""
        fail_node = Node()
        fail_node.is_fail_node = True
        A = self.compute_A()

        # add every character in A to the fail node pointing to its self
        for a in A:
            fail_node.out_edges[a] = fail_node
            fail_node.in_edges.append((a, fail_node))
        # add connections for the missing characters to the fail node for the whole trie
        self.add_missing_edges(fail_node, A)

    def add_missing_edges(self, fail_node, A):
        """add a connection to the fail node for all characters that the node misses"""
        if self.out_edges:  # if node is not final
            for a in A:
                if a not in self.out_edges.keys():
                    # create connections from this node to the fail node
                    fail_node.in_edges.append((a, self))
                    self.out_edges[a] = fail_node
            # apply the function recursively on all of the target nodes
            for char, target_node in self.out_edges.items():
                if not target_node.is_fail_node:
                    target_node.add_missing_edges(fail_node, A)

    def compute_A(self):
        """return a set of alphabets of the given trie"""
        self.reset_visited()
        A = set()
        self.collect_A(A)
        self.reset_visited()
        return A

    def collect_A(self, A):
        """traverse the trie and collect alphabets into set A"""
        if not self.visited:
            for char, out_node in self.out_edges.items():
                if out_node.is_fail_node:
                    continue
                A.add(char)
                out_node.collect_A(A)
            self.visited = True


class Hopcroft():

    def minimize(self, DEA):
        """minimize a given DEA using the Hopcroft Algorithm"""
        A = DEA.compute_A()
        F, NF = self.get_F_NF(DEA)
        P = self.hopcroft(A, NF, F)
        map = self.create_map(P)
        self.apply_hopcroft(DEA, map)

    def get_F_NF(self, trie):
        """compute and return F and NF of a given trie/DEA"""
        F = set()
        NF = set()
        self.compute_F_NF(trie, F, NF)
        return F, NF

    def compute_F_NF(self, node, F, NF):
        """traverse the trie/DEA and collect F and NF"""
        if not node.out_edges:  # if node has no out_edges, it is a final node
            F.add(node)
        else:
            NF.add(node)
            for _, target_node in node.out_edges.items():
                if node.is_fail_node:
                    NF.add(node)
                else:
                    self.compute_F_NF(target_node, F, NF)

    def split(self, X, a, Y):
        """split function of Hopcropft Algorithm"""
        X1 = set()
        X2 = set()
        for node in X:
            v = False
            for char, target_node in node.out_edges.items():
                if target_node in Y and char == a:
                    v = True
            if v:
                X1.add(node)
            else:
                X2.add(node)
        return X1, X2

    def minimum(self, sets):
        """given a set of sets, return the set that has the least elements"""
        sets = sorted(sets, key=len, reverse=False)
        for first in sets:
            return first

    def hopcroft(self, A, NF, F):
        """apply Hopcroft algorithm and return a set of equivalent classes of nodes"""
        P = set([frozenset(NF), frozenset(F)])
        W = {self.minimum(P)}
        while W != set():
            Y = W.pop()
            for a in A:
                for X in P.copy():
                    X1, X2 = self.split(X, a, Y)
                    if X1 != set() and X2 != set():
                        X1 = frozenset(X1)
                        X2 = frozenset(X2)
                        P.remove(X)
                        P.add(X1)
                        P.add(X2)
                        if X in W:
                            W.remove(X)
                            W.add(X1)
                            W.add(X2)
                        else:
                            s = set([X1, X2])
                            X_ = self.minimum(s)
                            W.add(X_)

        return P

    def create_map(self, P):
        """given P, return a dictionary of nodes mapped to their representatives"""
        mapping = {}
        for equi in P:
            equi = [node for node in equi]  # convert set into list
            repre = equi[0]  # pick a random representative node
            for node in equi:
                mapping[node] = repre

        return mapping

    def apply_hopcroft(self, node, map):
        """replace every node with its representative"""
        repr = map[node]  # the representative node
        if not node.is_fail_node:
            if node is repr:  # if a node is its own representative, do not change anything
                for char, out_node in node.out_edges.items():
                    self.apply_hopcroft(out_node, map)
            else:  # if a node is not a representative, replace it with the representative
                for in_char, in_node in node.in_edges:
                    # move the incoming connections of the current node to its representative
                    repr.in_edges.append((in_char, in_node))
                    # for each incoming node, point its outgoing connections to the representative node
                    in_node.out_edges[in_char] = repr
                node.in_edges = []
                # for all outgoing nodes, remove the connection to the current node
                for char, out_node in node.out_edges.items():
                    for in_char, in_node in out_node.in_edges:
                        if (char, node) == (in_char, in_node):
                            out_node.in_edges.remove((char, in_node))

                    self.apply_hopcroft(out_node, map)


if __name__ == "__main__":
    # create a trie from the word list
    trie = Node()
    word_list = argv[1]
    with open(word_list, "r") as f:
        for line in f:
            trie.add_word(line.strip())
    # convert the trie into a DEA
    trie.trie_to_DEA()
    # minimize the DEA with Hopcroft and print out the minimized trie
    hopcroft = Hopcroft()
    hopcroft.minimize(trie)
    trie.print_all(print_fail_node=False)
