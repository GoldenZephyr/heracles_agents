import os
from typing import Literal, Optional, Union

import spark_dsg
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings


class HeraclesDsgInterface(BaseSettings):
    dsg_interface_type: Literal["heracles"]
    uri: str
    username: SecretStr = Field(alias="HERACLES_NEO4J_USERNAME")
    password: SecretStr = Field(alias="HERACLES_NEO4J_PASSWORD")


class InContextDsgInterfaceConfig(BaseModel):
    dsg_interface_type: Literal["in_context"]
    dsg_filepath: Optional[str] = None
    _dsg: PrivateAttr() = None

    @model_validator(mode="after")
    def load_dsg(self):
        self._dsg = spark_dsg.DynamicSceneGraph.load(
            os.path.expandvars(self.dsg_filepath)
        )
        return self

    def get_dsg(self):
        return self._dsg


class NoDsgInterface(BaseModel):
    dsg_interface_type: Literal["none"]


DsgInterfaceConfigType = Union[
    HeraclesDsgInterface, InContextDsgInterfaceConfig, NoDsgInterface
]
