from lark import Lark

with open("cypher.lark", "r") as fo:
    cypher_bnf = fo.read()

parser = Lark(cypher_bnf, start="query")
tree = parser.parse('MATCH (p: Place {nodeSymbol: "P32"}) RETURN p')

test1 = """
MATCH (p:Person {name: "Alice"})-[:KNOWS]->(f:Person)
WHERE p.age > 30 AND NOT f.city = "London"
RETURN f.name, f.age
ORDER BY f.age DESC
LIMIT 5
"""
tree1 = parser.parse(test1)

test2 = """
MATCH (a)-[:FRIEND]->(b)-[:LIKES]->(c:Product {price: 20})
RETURN a.name, c.name
UNION ALL
MATCH (x:User)-[:BOUGHT]->(y:Product)
RETURN x.name, y.name
"""
tree2 = parser.parse(test2)
