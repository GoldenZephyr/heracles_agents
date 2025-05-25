# Heracles NLP<->DSG Experiments

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
