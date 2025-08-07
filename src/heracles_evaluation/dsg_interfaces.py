from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings
from typing import Literal, Union


class HeraclesDsgInterface(BaseSettings):
    dsg_interface_type: Literal["heracles"]
    uri: str
    username: SecretStr = Field(alias="HERACLES_NEO4J_USERNAME")
    password: SecretStr = Field(alias="HERACLES_NEO4J_PASSWORD")


class InContextDsgInterfaceConfig(BaseModel):
    dsg_interface_type: Literal["in_context"]
    an_example_field: int = 2


DsgInterfaceConfigType = Union[HeraclesDsgInterface, InContextDsgInterfaceConfig]
