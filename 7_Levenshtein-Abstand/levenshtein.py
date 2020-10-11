from sys import argv


def get_matrix_distance(source, target):
    source = " " + source
    target = " " + target

    d = [[0 for _ in range(len(target))] for _ in range(len(source))]

    for i in range(1, len(source)):
        d[i][0] = i
        for k in range(1, len(target)):
            d[0][k] = k
            cond = 1 if source[i] != target[k] else 0
            replace_op = d[i - 1][k - 1] + cond
            delete_op = d[i - 1][k] + 1
            insert_op = d[i][k - 1] + 1
            d[i][k] = min(replace_op, delete_op, insert_op)

    return d, d[len(source) - 1][len(target) - 1]


def print_matrix(d):
    for l in d:
        print(l)


def get_best_alignment(d, source, target):
    source = " " + source
    target = " " + target
    trace = []
    i = len(source) - 1
    k = len(target) - 1
    while i > 0 and k > 0:
        op = dict()  # dictionary of possible operations and their distances
        op["replace"] = d[i - 1][k - 1]
        op["del"] = d[i - 1][k]
        op["insert"] = d[i][k - 1]
        sorted_op = sorted(op.items(), key=lambda x: x[1])
        next_op, _ = sorted_op[0]
        if next_op == "replace":
            trace.append(source[i] + ":" + target[k])
            i = i - 1
            k = k - 1
        elif next_op == "del":
            trace.append(source[i] + ":" + " ")
            i = i - 1
            k = k
        else:
            trace.append(" " + ":" + target[k])
            i = i
            k = k - 1
    trace.reverse()
    return trace


if __name__ == "__main__":
    word_pair = argv[1]
    with open(word_pair) as f:
        for line in f:
            source, target = line.split()
            d, distance = get_matrix_distance(source, target)
            print_matrix(d)
            alignment = get_best_alignment(d, source, target)
            print("{} {}, distance = {}, alignment = ".format(source, target, distance), end=" ")
            print(*alignment)
