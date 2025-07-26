from pydantic import BaseModel
from typing import Literal, Union

class CypherDsgInterfaceConfig(BaseModel):
    dsg_interface_type: Literal["cypher"]
    an_example_field: int = 1 


class InContextDsgInterfaceConfig(BaseModel):
    dsg_interface_type: Literal["in_context"]
    an_example_field: int = 2 


DsgInterfaceConfigType = Union[CypherDsgInterfaceConfig, InContextDsgInterfaceConfig]
