import logging
import math

import spark_dsg

from heracles_evaluation.dsg_interfaces import PythonDsgInterface
from heracles_evaluation.tool_interface import FunctionParameter, ToolDescription
from heracles_evaluation.tool_registry import ToolRegistry, register_tool
from heracles_evaluation.tools.timeouts import FunctionTimeoutError, run_with_timeout

logger = logging.getLogger(__name__)


def execute_generated_code_timed(
    python_code: str, dsg_interface: PythonDsgInterface = None
):
    try:
        result = run_with_timeout(
            execute_generated_code, args=(python_code, dsg_interface), timeout=60
        )
        logger.debug(f"code_timed result: {result}")
        return result
    except FunctionTimeoutError:
        return "Your code timed out. In 60 seconds."


def execute_generated_code(python_code: str, dsg_interface: PythonDsgInterface = None):
    # Extract code between <python>...</python> tags if present
    # python_code_match = re.search(r"<python>(.*?)</python>", python_code, re.DOTALL)
    # if not python_code_match:
    #     error_msg = "warning, No <python>...</python> tags found in the generated code."
    #     print(error_msg)
    #     return error_msg
    # code_to_execute = python_code_match.group(1).strip()

    try:
        local_scope = {}
        exec_globals = {
            "spark_dsg": spark_dsg,
            "math": math,
        }  # make spark dsg and math available inside the execution blcok

        # Execute the code, which defines the 'solve_task' function in the local scope (generated code)
        exec(python_code, exec_globals, local_scope)
        solve_task_func = local_scope.get("solve_task")
        if callable(solve_task_func):
            logger.info("Executing function 'solve_task'...")
            result = solve_task_func(dsg_interface.get_dsg())
            logger.info(f"Execution successful. Result: {result}")
            return result
        else:
            logger.info("'solve_task' function not found in the generated code.")
            return "'solve_task' function not found in the generated code."

    except Exception as e:
        logger.error(f"Error executing generated code: {e}", exc_info=True)
        return str(e)


codegen_tool = ToolDescription(
    name="codegen_execute",
    description="A tool for executing Python code on a 3D Scene graph. The code takes in a spark_dsg graph object G",
    parameters=[
        FunctionParameter("python_code", str, "Python code to execute"),
    ],
    function=execute_generated_code,
)

register_tool(codegen_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
