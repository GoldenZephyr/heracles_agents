import os
from typing import Literal, Optional, Union

import spark_dsg
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings

from .pipelines.codegen_utils import load_dsg, load_dsg_api_prompt


class HeraclesDsgInterface(BaseSettings):
    dsg_interface_type: Literal["heracles"]
    uri: str
    # optional number of objects to check that the right dsg is loaded
    n_object_verification: Optional[int] = None
    username: SecretStr = Field(alias="HERACLES_NEO4J_USERNAME", exclude=True)
    password: SecretStr = Field(alias="HERACLES_NEO4J_PASSWORD", exclude=True)


class InContextDsgInterfaceConfig(BaseModel):
    dsg_interface_type: Literal["in_context"]
    dsg_filepath: Optional[str] = None
    dsg_place_layer_name: Optional[str] = None
    _dsg: PrivateAttr() = None

    @model_validator(mode="after")
    def load_dsg(self):
        self._dsg = spark_dsg.DynamicSceneGraph.load(
            os.path.expandvars(self.dsg_filepath)
        )
        return self

    def get_dsg(self):
        return self._dsg

    def get_place_layer_name(self):
        return self.dsg_place_layer_name


class NoDsgInterface(BaseModel):
    dsg_interface_type: Literal["none"]


class PythonDsgInterface(BaseModel):
    dsg_interface_type: Literal["python"]
    dsg_filepath: str
    dsg_labels_filepath: Optional[str] = None
    dsg_api_filepath: str
    dsg_api_descriptions: Optional[bool] = False
    dsg_api_examples: Optional[bool] = False

    _dsg: PrivateAttr() = None

    @model_validator(mode="after")
    def load_dsg(self):
        self._dsg = load_dsg(
            os.path.expandvars(self.dsg_filepath),
            os.path.expandvars(self.dsg_labels_filepath)
            if self.dsg_labels_filepath
            else None,
        )
        return self

    @model_validator(mode="after")
    def load_dsg_api(self):
        self._dsg_api_prompt = load_dsg_api_prompt(
            os.path.expandvars(self.dsg_api_filepath),
            include_descriptions=self.dsg_api_descriptions,
            include_examples=self.dsg_api_examples,
        )
        return self

    def get_dsg(self):
        return self._dsg

    def get_dsg_api_prompt(self):
        return self._dsg_api_prompt


DsgInterfaceConfigType = Union[
    HeraclesDsgInterface,
    InContextDsgInterfaceConfig,
    NoDsgInterface,
    PythonDsgInterface,
]
