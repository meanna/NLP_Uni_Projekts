from sys import argv

str2int = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "0": 0}
symbols = ["*", "/", "-", "+", "(", ")", "."]


def integer(exp, start):
    try:
        result = str2int[exp[start]]
    except:
        raise Exception("error at position ", start, "input should be numbers and arithmetic operations")
    else:
        start += 1
        while start < len(exp):
            if exp[start] in str2int.keys():
                result = (result * 10) + str2int[exp[start]]
                start += 1
            elif exp[start] not in symbols:
                raise Exception("error at position ", start, "input should be numbers and arithmetic operations")
            else:
                return result, start
        return result, start


def number(exp, start):
    num, start = integer(exp, start)
    if start < len(exp):
        if exp[start] != ".":
            return num, start
        else:
            num2, end = integer(exp, start + 1)
            num = num + num2 * (10 ** (-(end - start - 1)))
            return num, end
    else:
        return num, start


def mul(exp, start):
    num, pos = bracketexpr(exp, start)
    if len(exp) > pos:
        if exp[pos] == "*":
            num2, end = mul(exp, pos + 1)
            return (num * num2), end

        elif exp[pos] == "/":
            num2, end = mul(exp, pos + 1)
            return (num / num2), end
        else:
            return num, pos
    else:
        return num, pos


def bracketexpr(exp, start):
    if exp[start] == "(":
        num, end = add(exp, start + 1)
        if end < len(exp):
            if exp[end] != ")":
                raise Exception("missing a closing parenthesis at position " + str(end))
            else:
                return num, end + 1
        else:
            raise Exception("missing a closing parenthesis at position " + str(end))
    elif exp[start] == "-":
        num, end = number(exp, start + 1)
        return num * (-1), end
    else:
        return number(exp, start)


def add(exp, start):
    num, pos = mul(exp, start)
    if len(exp) > pos:
        if exp[pos] == "+":
            num2, end = add(exp, pos + 1)
            return num + num2, end
        elif exp[pos] == "-":
            num2, end = add(exp, pos + 1)
            return num - num2, end
        else:
            return num, pos
    else:
        return num, pos


def calculate(exp):
    # filter spaces out of the input
    no_space = [i for i in exp if i != " "]
    if no_space:
        result, _ = add(no_space, 0)
        print(result)


if __name__ == "__main__":
    file = argv[1]
    with open(file,"r") as f:
        for line in f:
            calculate(line.strip())