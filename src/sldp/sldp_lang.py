from sldp.hand_parser import parse_sldp  # NOQA
from sldp.lark_parser import lark_parse_sldp  # NOQA

# sldp_parser_impl = parse_sldp
sldp_parser_impl = lark_parse_sldp


def get_sldp_type(s: str):
    ast = sldp_parser_impl(s)
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
    elif type(ast) is str:
        return "string"

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
            return a.strip().lower() == b.strip().lower()

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


def sldp_equals(s1, s2):
    exp1 = sldp_parser_impl(s1)
    exp2 = sldp_parser_impl(s2)

    return equals(exp1, exp2)


if __name__ == "__main__":
    a = "[1, 2, 3]"
    print(sldp_parser_impl(a))

    b = "[1, 2, <1, 2, 3>]"
    print(sldp_parser_impl(b))

    c = "[1, 2, POINT(2.3 1 2)]"
    print(sldp_parser_impl(c))

    d = "{k1: v1, k2: v2}"
    print(sldp_parser_impl(d))

    e = "[{k0: 1.12}, {k1: v1, k2: v2}]"
    print(sldp_parser_impl(e))

    f = "[{k0: 1.12}, {k1: v1, k2: POINT(1.12 2 3)}]"
    print(sldp_parser_impl(f))

    solution = "{tree: 163, fence: 17, vehicle: 26, seating: 9, window: 1, sign: 6, pole: 21, door: 3, box: 4, trash: 1, rock: 62, bag: 1}"
    answer = "{tree: 163, rock: 62, pole: 21, vehicle: 26, box: 4, fence: 17, seating: 9, window: 1, sign: 6, door: 3, trash: 1, bag: 1}"

    print(sldp_equals(solution, answer))
