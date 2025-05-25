def get_sldp_type(s: str):
    ast = parse_sldp(s)
    if type(ast) is tuple:
        if is_list(ast):
            return "list"
        elif is_dict(ast):
            return "dict"
        elif is_set(ast):
            return "set"
        elif is_point(ast):
            return "point"
    elif type(ast) is float:
        return "number"

    raise Exception(f"Unknown type for {s}")


def is_list(t: tuple):
    return t[0] == "list"


def is_dict(t: tuple):
    return t[0] == "dict"


def is_set(t: tuple):
    return t[0] == "set"


def is_point(t: tuple):
    return t[0] == "point"


def equals(a, b):
    if type(a) is tuple and type(b) is tuple:
        if is_list(a) and is_list(b):
            return list_equals(a, b)
        elif is_dict(a) and is_dict(b):
            return dict_equals(a, b)
        elif is_set(a) and is_set(b):
            return set_equals(a, b)
        elif is_point(a) and is_point(b):
            return point_equals(a, b)
    else:
        if type(a) is float and type(b) is float:
            return float_equals(a, b)

        if type(a) is str and type(b) is str:
            return a.strip() == b.strip()

    return False


def list_equals(a: tuple, b: tuple):
    """List is like ("list", e1, e2, e4)"""
    if len(a) != len(b):
        return False
    return all([equals(e1, e2) for e1, e2 in zip(a, b)])


def extract_uniform_keys(list_of_dicts):
    keys = set()
    for _, k, v in list_of_dicts[0]:
        keys.add(k)

    for d in list_of_dicts[1:]:
        these_keys = set()
        for _, k, v in d:
            these_keys.add(k)

        if these_keys != keys:
            raise Exception("Keys are not uniform")

    return keys


def dict_lookup(d, key):
    for _, k, v in d[1:]:
        if equals(k, key):
            return v
    return None


def dict_equals(a: tuple, b: tuple):
    """Dict is like ("dict", ("pair", k1, v1), ("pair", k2, v2))"""
    assert is_dict(a)
    assert is_dict(b)
    if len(a) != len(b):
        return False

    # Every key in a is found in b (and value matches)
    for _, ka, va in a[1:]:
        vb = dict_lookup(b, ka)
        if vb is None:
            print(f"{ka} from a not in b")
            return False
        if not equals(va, vb):
            print(f"For key {ka}, value in a ({va}) does not match value in b ({vb})")
            return False

    # Every key in b is found in a (and value matches)
    for _, kb, vb in b[1:]:
        va = dict_lookup(a, kb)
        if vb is None:
            print(f"{kb} from b not in a")
            return False
        if not equals(va, vb):
            return False

    return True


def element_in_set(a, b: tuple):
    assert is_set(b)
    for e in b[1:]:
        if equals(a, e):
            return True
    return False


def set_equals(a: tuple, b: tuple):
    assert is_set(a)
    assert is_set(b)

    for e in a[1:]:
        if not element_in_set(e, b):
            return False

    for e in b[1:]:
        if not element_in_set(e, a):
            return False

    return True


def float_equals(a: float, b: float):
    return abs(a - b) < 0.01


def point_equals(a: tuple, b: tuple):
    """Currently L_inf equality check"""
    assert is_point(a)
    assert is_point(b)

    if len(a) != len(b):
        return False

    for ca, cb in zip(a[1:], b[1:]):
        if not float_equals(ca, cb):
            return False
    return True


def tokenize_sldp(string):
    string = string.replace("\n", "")

    delimiters = ["[", "]", "{", "}", "<", ">", "(", ")", ":", ","]

    for d in delimiters:
        string = string.replace(d, " " + d + " ")

    return tuple(string.split())


def parse_list(toks):
    return parse_collection("[", "]", ",", "list", toks)


def parse_set(toks):
    return parse_collection("<", ">", ",", "set", toks)


def parse_collection(open_delim, close_delim, sep, name, toks):
    assert toks[0] == open_delim

    toks = toks[1:]
    data = (name,)
    while len(toks) > 0:
        next_datum, toks = parse(toks)
        data += (next_datum,)
        if toks[0] == close_delim:
            break
        if toks[0] != sep:
            raise Exception("Malformed syntax! Expected ,")
        toks = toks[1:]
    toks = toks[1:]
    return data, toks


def parse_point(toks):
    """Should be of the form POINT(x y z)"""
    assert toks[0] == "POINT"

    data = ("point",)
    if toks[1] != "(":
        raise Exception("Expected ( after POINT")

    data += (float(toks[2]),)
    data += (float(toks[3]),)
    data += (float(toks[4]),)

    if toks[5] != ")":
        raise Exception("Expected closing ) after POINT")

    return data, toks[6:]


def parse_dict(toks):
    """should be of the form {k1: v1, k2: v2}"""

    assert toks[0] == "{"

    data = ("dict",)

    toks = toks[1:]
    while len(toks) > 0:
        next_key, toks = parse(toks)
        if toks[0] != ":":
            raise Exception("Invalid dictionary, expected :")
        next_value, toks = parse(toks[1:])
        data += (("pair", next_key, next_value),)
        if toks[0] == "}":
            break
        if toks[0] != ",":
            raise Exception("Invalid dictionary, expected , between entries")
        toks = toks[1:]

    toks = toks[1:]
    return data, toks


def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def parse(toks):
    # print("Parsing: ", toks)
    if type(toks) is tuple:
        match toks[0]:
            case "[":
                return parse_list(toks)
            case "<":
                return parse_set(toks)
            case "{":
                return parse_dict(toks)
            case "POINT":
                return parse_point(toks)
            case _:
                if is_float(toks[0]):
                    return float(toks[0]), toks[1:]
                else:
                    return toks[0], toks[1:]
    else:
        if is_float(toks):
            return float(toks), []
        else:
            return toks, []


def parse_sldp(string):
    """
    [a, b, c] - list
    {a: 1, b: 2} - dict
    <1, 2, 3> - set
    TYPENAME() - type
    """
    toks = tokenize_sldp(string)

    ast, extra_toks = parse(toks)
    if len(extra_toks) > 0:
        raise Exception(f"Malformed input, found extra tokens: {extra_toks}")
    return ast


def sldp_equals(s1, s2):
    exp1 = parse_sldp(s1)
    exp2 = parse_sldp(s2)

    return equals(exp1, exp2)


if __name__ == "__main__":
    a = "[1, 2, 3]"
    print(parse_sldp(a))

    b = "[1, 2, <1, 2, 3>]"
    print(parse_sldp(b))

    c = "[1, 2, POINT(2.3 1 2)]"
    print(parse_sldp(c))

    d = "{k1: v1, k2: v2}"
    print(parse_sldp(d))

    e = "[{k0: 1.12}, {k1: v1, k2: v2}]"
    print(parse_sldp(e))

    f = "[{k0: 1.12}, {k1: v1, k2: POINT(1.12 2 3)}]"
    print(parse_sldp(f))

    solution = "{tree: 163, fence: 17, vehicle: 26, seating: 9, window: 1, sign: 6, pole: 21, door: 3, box: 4, trash: 1, rock: 62, bag: 1}"
    answer = "{tree: 163, rock: 62, pole: 21, vehicle: 26, box: 4, fence: 17, seating: 9, window: 1, sign: 6, door: 3, trash: 1, bag: 1}"

    print(sldp_equals(solution, answer))
