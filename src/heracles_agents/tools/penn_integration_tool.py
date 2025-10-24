import time

import numpy as np
import zmq
from pydantic import BaseModel, ConfigDict, field_validator

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool

context = zmq.Context()


class PennQuadCommand(BaseModel):
    x: float
    y: float
    timestamp: int
    query: str = ""


class UtmToMapInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    local_utm_origin: np.ndarray
    map_offset: np.ndarray

    @field_validator("local_utm_origin", "map_offset", mode="before")
    def convert_to_array(cls, v):
        return np.array(v)


def send_waypoint_to_quad(
    northing,
    easting,
    zone: str = "18N",
    zmq_uri: str = None,
    utm_map_info: UtmToMapInfo = None,
):
    if zone != "18N":
        raise ValueError("Currently only waypoints in zone 18N are supported")

    position_utm = np.array([easting, northing])
    pos_rel = position_utm - utm_map_info.local_utm_origin + utm_map_info.map_offset

    cmd = PennQuadCommand(x=pos_rel[0], y=pos_rel[1], timestamp=int(time.time() * 1e9))

    data_to_send = cmd.model_dump()
    print(f"Sending cmd: {data_to_send} to penn quadrotor")
    socket = context.socket(zmq.PUSH)  # Why is this PUSH and not PUB?
    try:
        socket.bind(zmq_uri)  # Why is this bind and not connect?
        socket.send_pyobj(cmd.model_dump())
    except Exception as ex:
        return str(ex)
    finally:
        socket.close()

    return f"Sent goal {data_to_send}"


waypoint_tool = ToolDescription(
    name="send_waypoint_to_quad",
    description="An interface for sending a waypoint to a quadrotor.",
    parameters=[
        FunctionParameter(
            "northing", float, "The northing value of the quadrotor UTM waypoint."
        ),
        FunctionParameter(
            "easting", float, "The easting value of the quadrotor UTM waypoint."
        ),
    ],
    function=send_waypoint_to_quad,
)

register_tool(waypoint_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
