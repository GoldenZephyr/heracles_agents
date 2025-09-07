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

    # Handle empty
    if len(toks) > 0 and toks[0] == close_delim:
        toks = toks[1:]
        return data, toks

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

    # Handle empty
    if len(toks) > 0 and toks[0] == "}":
        toks = toks[1:]
        return data, toks

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
