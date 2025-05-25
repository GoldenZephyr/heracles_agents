# Heracles NLP<->DSG Experiments

## Pipeline

The evaluation pipeline assumes an initial scene graph has been loaded to
heracles. The queries that get evaluated are defined in
`evaluation_questions.yaml`. Each question is defined with an associated
answer.  Then, the script `chatgpt_benchmark.py` is run to create a cypher
query for answering each question. This results in intermediate answers
including the output of the cypher query when run against the database.  There
is still some ambiguity in comparing these cypher query results to the correct
solution, so there is an "answer refinement" step run with `refine_answers.py`,
which prompts chatgpt with the original question and the results of the cypher
query. This script then compares the refined answer to the solution, using
the SLDP equality language (see below) to evaluate equality. The final
results can be summarized by running `results_table.py`.

## SLDP Equality Language

In order to evaluate whether the LLM pipeline is correct, we need to define
a sense of equality and implement a method for checking it. This is rather
tricky, because there are different senses in which things can be equal.

We need to be able to handle Lists, Sets, Dictionaries, and Points.
Lists are equal if each element is equal, sets A and B are equal if A ⊆ B and
B ⊆ A, Dictionaries are equal if the sets of their keys are equal and the value
for each key matches between dictionaries, and two points are equal if they
are  within some tolerance. Of course primitive numbers and strings can also
be compared for equality. We want to support arbitrary compositions of these
containers.

### Syntax

A `list` is written as `[element1, element2, ... elementN]`

A `set` is written as `<element1, element2, ... elementN>`

A `dict` is written as `{k1: v1, k2: v2}`

A `point` is written as `POINT(x y z)` (note the lack of comma)

### Interface

There are two relevant functions for interacting with the SLDP language.  First
is `parse_sldp`. If you call `parse_sldp` on your string, you can verify
whether it correctly parses into the AST.  Second is `sldp_equals(s1, s2)`
which will return true if sldp strings s1 and s2 are equal in the sense of
sldp.
