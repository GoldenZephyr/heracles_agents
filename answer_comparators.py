def convert_to_number(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a number, 
        Your response should contain only the number
        and no extraneous information.
        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


def convert_to_list(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a list of the form [element1, element2, .... elementN], 
        maintaining the order implied by the intermediate data. 
        Elements within the list should not have quotations around them. 
        If you need to represent a POINT in the list, the syntax is POINT(x y z).
        Your response should contain only the list and no extraneous information.

        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


def convert_to_set_of_dicts(query_interface_fn, question, answer, keys):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a set of dictionaries of the form < {{key1: value1, ..., keyN: valueN}}, {{key3: value3}}, ... >
        Keys and values should not have quotations around them. Each dictionary should have these keys: {keys}.
        If you need to represent a POINT, the syntax is POINT(x y z).
        Your response should contain only the list of dictionaries and no extraneous information.
        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


def convert_to_dict(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a dictionaries of the form {{key1: value1, ..., keyN: valueN}}
        Keys and values should not have quotations around them.
        If you need to represent a POINT, the syntax is POINT(x y z).
        Your response should contain only the dictionary and no extraneous information.
        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


def convert_to_set(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a set of the form <element1, element2, .... elementN>.
        Elements within the set should not have quotations around them. 
        If you need to represent a POINT, the syntax is POINT(x y z).
        Your response should contain only the set <element1, ..., elementN>, 
        and no extraneous information.

        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response
