import os

import openai

from heracles_evaluation.prompt import get_sldp_format_description
from sldp.lark_parser import get_sldp_lark_grammar

key = os.getenv("DSG_OPENAI_API_KEY")

client = openai.OpenAI(
    api_key=key,
    timeout=60,
)

sldp_description = get_sldp_format_description()

test_input = f"I have a collection of an apple, an orange and a pear. Call the lark_answer tool with an SLDP-formatted answer. \n{sldp_description}"

lark_grammar = get_sldp_lark_grammar()

tools = [
    {
        "type": "custom",
        "name": "lark_answer",
        "description": "Call this tool to submit your answer to the question",
        "format": {"type": "grammar", "syntax": "lark", "definition": lark_grammar},
    }
]
response = client.responses.create(
    model="gpt-5-mini",
    input=test_input,
    text={"format": {"type": "text"}},
    tools=tools,
    parallel_tool_calls=False,
    max_output_tokens=3000,
)

print("Response:\n\n")
print(response.output)
