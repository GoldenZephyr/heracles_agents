import os
import time

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool


def visualize_objects(objects: str, x: str, y: str, clear: bool, viz_topic: str = None):
    if viz_topic is None:
        raise ValueError("visualize_objects called with viz_topic missing")

    if clear:
        # Clear all markers in the 'objects' namespace
        clear_cmd = f"""
        ros2 topic pub {viz_topic} visualization_msgs/msg/MarkerArray "{{
            markers: [{{
                header: {{
                    frame_id: 'map'
                }},
                ns: 'objects',
                action: 3
            }}]
        }}" -1
        """
        os.system(clear_cmd)
        time.sleep(0.1)
        return "Successfully cleared the visualization."

    object_list = [s.strip() for s in objects.split(",")]
    x_list = [float(s.strip()) for s in x.split(",")]
    y_list = [float(s.strip()) for s in y.split(",")]

    if not (len(object_list) == len(x_list) == len(y_list)):
        return f"Failed to visualize the following objects {object_list}. Number of coordinates and objects must match."

    # Build marker array
    markers = []
    for i, name in enumerate(object_list):
        marker = f"""{{
            header: {{
                frame_id: 'map'
            }},
            ns: 'objects',
            id: {i},
            type: 3,
            action: 0,
            pose: {{
                position: {{x: {x_list[i]}, y: {y_list[i]}, z: 6.0}},
                orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
            }},
            scale: {{x: 0.2, y: 0.2, z: 20.0}},
            color: {{r: 1.0, g: 1.0, b: 1.0, a: 1.0}},
            lifetime: {{sec: 0, nanosec: 0}}
        }}"""
        markers.append(marker)

    markers_str = ",\n".join(markers)

    # First clear existing markers
    clear_cmd = f"""
    ros2 topic pub {viz_topic} visualization_msgs/msg/MarkerArray "{{
        markers: [{{
            header: {{
                frame_id: 'map'
            }},
            ns: 'objects',
            action: 3
        }}]
    }}" -1
    """
    os.system(clear_cmd)
    time.sleep(0.1)

    # Publish all markers at once
    cmd = f"""
    ros2 topic pub {viz_topic} visualization_msgs/msg/MarkerArray "{{
        markers: [
            {markers_str}
        ]
    }}" -1
    """
    os.system(cmd)

    return f"Visualized the following objects in RViz: {object_list}."


viz_objects_tool = ToolDescription(
    name="visualize_objects",
    description="An interface for visualizing scene graph objects on RViz. Use this to visualize node symbols at given locations to RViz for the user. Also use this to clear the visualization.",
    parameters=[
        FunctionParameter(
            "objects", str, "A comma-separated list of node symbols visualize."
        ),
        FunctionParameter(
            "x",
            str,
            "A comma-separated list x-coordinates for the corresponding node symbol. The number of node symbols must match the number of x-coordinates.",
        ),
        FunctionParameter(
            "y",
            str,
            "A comma-separated list y-coordinates for the corresponding node symbol. The number of node symbols must match the number of y-coordinates.",
        ),
        FunctionParameter(
            "clear",
            bool,
            "If True, it will clear the visualization and do nothing else. If False, it will visualize the given objects.",
        ),
    ],
    function=visualize_objects,
)

register_tool(viz_objects_tool)
print("Registered tools:")
print(ToolRegistry.registered_tool_summary())
