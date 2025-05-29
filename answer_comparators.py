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


def convert_to_string(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a string, 
        Your response should contain only a *single* word, no quotation,
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
        The list is denoted by square brackets [ ]. 
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
        The set is denoted by angle brackets < >, and each dictionary is denoted by curly braces {{ }}.
        If you need to represent a POINT, the syntax is POINT(x y z).
        Your response should contain only the list of dictionaries and no extraneous information.
        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


def convert_to_dict(query_interface_fn, question, answer):
    prompt = f""" You are trying to answer the question: {question}.
        You have the following data as an intermediate answer. Please
        reformat the following data into a dictionary of the form {{key1: value1, ..., keyN: valueN}}
        Keys and values should not have quotations around them.
        The dictionary is denoted by curly braces {{ }}.
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
        The set is denoted by angle brackets < >.
        If you need to represent a POINT, the syntax is POINT(x y z).
        Your response should contain only the set <element1, ..., elementN>, 
        and no extraneous information.

        """

    prompt += answer

    response = query_interface_fn(prompt)
    return response


# agent_set_answer_ps = """
#        Please format final_answer into a set of the form <element1, element2, .... elementN>.
#        Elements within the set should not have quotations around them.
#        The set is denoted by angle brackets < >. If you need to represent a POINT, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
#        final_answer should contain only the set <element1, ..., elementN>,
#        and no extraneous information."""
#
#
# agent_dict_answer_ps = """
#        Please format final_answer into a dictionary of the form {{key1: value1, ..., keyN: valueN}}
#        Keys and values should not have quotations around them.
#        The dictionary is denoted by curly braces {{ }}.
#        If you need to represent a POINT, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
#        final_answer should contain only the dictionary, and no extraneous information."""
#
# agent_number_answer_ps = """ Please format final_answer into a number.
#        Your response should contain only the number
#        and no extraneous information. """
#
# agent_string_answer_ps = """ Please format final_answer into a string,
#        Your final answer should contain only a *single* word, no quotation,
#        and no extraneous information.
#        """
#
# agent_list_answer_ps = """ Please format final_answer into a list of the form [element1, element2, .... elementN],
#        maintaining the order implied by the intermediate data.
#        The list is denoted by square brackets [ ].
#        Elements within the list should not have quotations around them.
#        If you need to represent a POINT in the list, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
#        final_answer should contain only the list and no extraneous information. """
#
#


agent_set_answer_ps = """
        Please format your final answer into a set of the form <element1, element2, .... elementN>.
        Elements within the set should not have quotations around them. 
        The set is denoted by angle brackets < >. If you need to represent a POINT, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
        The final answer should look like
        <answer>
        < element1, ..., elementN >
        </answer>"""


agent_dict_answer_ps = """
        Please format your final answer into a dictionary of the form {key1: value1, ..., keyN: valueN}
        Keys and values should not have quotations around them.
        The dictionary is denoted by curly braces { }.
        If you need to represent a POINT, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
        The final answer should look like
        <answer>
        {key1: value1, ..., keyN: valueN}
        </answer>"""

agent_number_answer_ps = """ Please format your final answer into a number.
        Your response should contain only the number, looking like
        <answer>123</answer>. """

agent_string_answer_ps = """ Please format your final answer into a string, 
        Your final answer should contain only a *single* word, no quotation,
        and no extraneous information, looking like 
        <answer>The answer string</answer>
        """

agent_list_answer_ps = """ Please format your final answer into a list of the form [element1, element2, .... elementN], 
        maintaining the order implied by the intermediate data.
        The list is denoted by square brackets [ ]. 
        Elements within the list should not have quotations around them. 
        If you need to represent a POINT in the list, the syntax is POINT(x y z) (only for answering, not for writing cypher queries).
        The final answer should look like
        <answer>
        [element1, ..., elementN]
        </answer>. """
