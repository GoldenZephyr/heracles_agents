import argparse
import logging

import spark_dsg

from heracles_agents.pipelines.codegen_utils import load_dsg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    scene_graph = load_dsg(args.dsg_path, args.labelspace_path)

    result = solve_task(scene_graph)
    logging.info(f"Result: {result}")
    # result_object_list_all_places = solve_task_object_list_all_places(scene_graph)
    # # logging.info(f"Result for all places: {result_object_list_all_places}")
    # result_go_to_object_2 = solve_task_object_2(scene_graph)
    # logging.info(f"Result for object 2: {result_go_to_object_2}")
    # result_go_to_place_100 = solve_task_place_100(scene_graph)
    # logging.info(f"Result for place 100: {result_go_to_place_100}")
    # result_go_to_trash = solve_task_go_to_trash(scene_graph)
    # logging.info(f"Result for go to trash: {result_go_to_trash}")
    # result_go_to_all_trash = solve_task_go_to_all_trash(scene_graph)
    # logging.info(f"Result for go to all trash: {result_go_to_all_trash}")
    # result_trash_and_box = solve_task_trash_and_box(scene_graph)
    # logging.info(f"Result for trash and box: {result_trash_and_box}")
    # result_first_inspect_box_then_go_to_trash = solve_task_first_inspect_box_then_go_to_trash(scene_graph)
    # logging.info(f"Result for first inspect box then go to trash: {result_first_inspect_box_then_go_to_trash}")
    # result_walk_to_trash_or_inspect_box = solve_task_walk_to_trash_or_inspect_box(scene_graph)
    # logging.info(f"Result for walk to trash or inspect box: {result_walk_to_trash_or_inspect_box}")
    # result_move_to_each_pole = solve_task_move_to_each_pole(scene_graph)
    # logging.info(f"Result for move to each pole: {result_move_to_each_pole}")
    # result_print_place_semantics = solve_task_print_place_semantics(scene_graph)
    # logging.info(f"Result for print place semantics: {result_print_place_semantics}")
    # result_print_room_semantics = solve_task_print_room_semantics(scene_graph)
    # logging.info(f"Result for print room semantics: {result_print_room_semantics}")
    # result_print_available_layers = solve_task_print_available_layers(scene_graph)
    # logging.info(f"Result for print available layers: {result_print_available_layers}")
    # result_print_building_semantics = solve_task_print_building_semantics(scene_graph)
    # logging.info(f"Result for print building semantics: {result_print_building_semantics}")
    result_print_meshplace_semantics = solve_task_print_meshplace_semantics(scene_graph)
    logging.info(
        f"Result for print meshplace semantics: {result_print_meshplace_semantics}"
    )

    logging.info(f"MESH_PLACES ID is: {spark_dsg.DsgLayers.MESH_PLACES}")
    logging.info(f"Rooms ID is: {spark_dsg.DsgLayers.ROOMS}")
    logging.info(f"Buildings ID is: {spark_dsg.DsgLayers.BUILDINGS}")
    logging.info(f"Objects ID is: {spark_dsg.DsgLayers.OBJECTS}")
    layer_keys = [(layer.id, layer.partition) for layer in scene_graph.layers]
    logging.info(f"Available Layer Keys: {layer_keys}")


def solve_task(G):
    # Collect and return all object NodeSymbols in the graph
    objects_layer = G.get_layer(spark_dsg.DsgLayers.OBJECTS)
    object_ids = [node.id for node in objects_layer.nodes]
    return object_ids


def solve_task_object_list_all_places(G):
    # Collect and return all place NodeSymbols in the graph
    places_layer = G.get_layer(spark_dsg.DsgLayers.PLACES)
    place_ids = [node.id for node in places_layer.nodes]
    return place_ids


def solve_task_object_2(G):
    # Return the NodeSymbol for object 2 (category 'O', index 2)
    obj_id = spark_dsg.NodeSymbol("O", 2)
    if G.has_node(obj_id):
        return obj_id
    return None


def solve_task_place_100(G):
    """
    Returns the NodeSymbol for place 100 if it exists, otherwise None.
    """
    place_symbol = spark_dsg.NodeSymbol("P", 100)
    if G.has_node(place_symbol):
        return place_symbol

    return None


def solve_task_go_to_trash(G):
    """
    Task: Move over to the trash
    Returns the NodeSymbols of all nodes labeled 'trash'.
    """

    labelspace = G.metadata.get()["labelspace"]
    trash_nodes = []

    # Look through all objects and check their semantic label
    for obj in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        label_id = str(obj.attributes.semantic_label)
        if label_id in labelspace and "trash" in labelspace[label_id].lower():
            trash_nodes.append(obj.id)

    return trash_nodes


def solve_task_go_to_all_trash(G):
    """
    Task: Move over to all trashes
    Returns a list of NodeSymbols for all objects labeled 'trash'.
    """

    labelspace = G.metadata.get()["labelspace"]
    trash_nodes = []

    # Look through all objects and check their semantic label
    for obj in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        label_id = str(obj.attributes.semantic_label)
        if label_id in labelspace and "trash" in labelspace[label_id].lower():
            trash_nodes.append(obj.id)

    return trash_nodes


def solve_task_trash_and_box(G):
    """
    Task: Drive over to the trash and inspect the box.
    Returns two lists of NodeSymbols: one for trash nodes and one for box nodes.
    """

    labelspace = G.metadata.get()["labelspace"]
    trash_nodes = []
    box_nodes = []

    # Iterate over all objects in the graph
    for obj in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        label_id = str(obj.attributes.semantic_label)
        if label_id not in labelspace:
            continue

        label = labelspace[label_id].lower()
        if "trash" in label:
            trash_nodes.append(obj.id)
        elif "box" in label:
            box_nodes.append(obj.id)

    return {"trash": trash_nodes, "box": box_nodes}


def solve_task_first_inspect_box_then_go_to_trash(G):
    """
    Task: First inspect the box and then head over to the trash.
    Returns two ordered lists of NodeSymbols: first for boxes, then for trash nodes.
    """

    labelspace = G.metadata.get()["labelspace"]
    box_nodes = []
    trash_nodes = []

    # Iterate over all objects in the graph
    for obj in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        label_id = str(obj.attributes.semantic_label)
        if label_id not in labelspace:
            continue

        label = labelspace[label_id].lower()
        if "box" in label:
            box_nodes.append(obj.id)
        elif "trash" in label:
            trash_nodes.append(obj.id)

    # Order: first boxes, then trash
    return {"box": box_nodes, "trash": trash_nodes}


def solve_task_walk_to_trash_or_inspect_box(G):
    """
    Task: Walk over to the trash or inspect the box.
    Returns a dictionary with NodeSymbols for either trash or box nodes.
    Prioritizes trash nodes if any exist, otherwise returns box nodes.
    """

    """
    Task: Return all options for 'trash' and 'box'.
    Returns a dictionary with NodeSymbols for both trash and box nodes.
    """

    labelspace = G.metadata.get()["labelspace"]
    trash_nodes = []
    box_nodes = []

    # Iterate over all objects in the graph
    for obj in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        label_id = str(obj.attributes.semantic_label)
        if label_id not in labelspace:
            continue

        label = labelspace[label_id].lower()
        if "trash" in label:
            trash_nodes.append(obj.id)
        elif "box" in label:
            box_nodes.append(obj.id)

    return {"trash": trash_nodes, "box": box_nodes}


def solve_task_move_to_each_pole(G):
    """
    Task: Move to each pole in the sidewalk room.
    Returns a list of NodeSymbols for all poles located in the 'sidewalk' room.
    """

    labelspace = G.metadata.get()["labelspace"]
    logging.info(f"Labelspace: {labelspace}")
    poles_in_sidewalk = []

    # Iterate over all places to find those in the 'sidewalk' room
    for place in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes:
        parent_room_id = place.get_parent()
        if parent_room_id is None:
            continue

        parent_room = G.get_node(parent_room_id)
        room_label_id = str(parent_room.attributes.semantic_label)
        if room_label_id not in labelspace:
            continue

        if "sidewalk" in labelspace[room_label_id].lower():
            # Find all objects in this place
            for obj_id in place.children():
                obj_node = G.get_node(obj_id)
                obj_label_id = str(obj_node.attributes.semantic_label)
                if (
                    obj_label_id in labelspace
                    and "pole" in labelspace[obj_label_id].lower()
                ):
                    poles_in_sidewalk.append(obj_node.id)

    return poles_in_sidewalk


def solve_task_print_place_semantics(G):
    """
    Task: Print all semantics associated with places.
    Returns a dictionary mapping each place NodeSymbol to its semantic label,
    excluding places with unknown labels.
    """

    labelspace = G.metadata.get()["labelspace"]
    place_semantics = {}

    # Iterate over all places
    for place in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes:
        label_id = str(place.attributes.semantic_label)
        label = labelspace.get(label_id)
        if label is not None:
            place_semantics[place.id] = label

    return place_semantics


def solve_task_print_room_semantics(G):
    """
    Task: Print all semantics associated with rooms.
    Returns a dictionary mapping each room NodeSymbol to its semantic label.
    """

    labelspace = G.metadata.get()["labelspace"]
    room_semantics = {}

    # Iterate over all rooms
    for room in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        label_id = str(room.attributes.semantic_label)
        label = labelspace.get(label_id, "Unknown")
        room_semantics[room.id] = label

    return room_semantics


def solve_task_print_available_layers(G):
    """
    Task: Print all available non-partitioned layers in the graph.
    Returns a list of tuples with (layer_id, number of nodes).
    """

    layers_info = []

    for layer in G.layers:
        layers_info.append((layer.id, layer.num_nodes()))

    return layers_info


def solve_task_print_building_semantics(G):
    """
    Task: Print all semantics associated with buildings.
    Returns a dictionary mapping each building NodeSymbol to its semantic label.
    """

    labelspace = G.metadata.get()["labelspace"]
    building_semantics = {}

    # Iterate over all buildings
    for building in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes:
        label_id = str(building.attributes.semantic_label)
        label = labelspace.get(label_id, "Unknown")
        building_semantics[building.id] = label

    return building_semantics


def solve_task_print_meshplace_semantics(G):
    """
    Task: Print all semantics associated with MeshPlaces.
    Returns a dictionary mapping each MeshPlace NodeSymbol to its semantic label.
    """

    labelspace = G.metadata.get()["labelspace"]
    meshplace_semantics = {}

    # Iterate over all MeshPlaces
    for meshplace in G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes:
        label_id = str(meshplace.attributes.semantic_label)
        label = labelspace.get(label_id, "Unknown")
        meshplace_semantics[meshplace.id] = label

    return meshplace_semantics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsg_path",
        default="/home/harel/code/heracles_agents/examples/scene_graphs/west_point_fused_map_wregions_labelspace_resaved.json",
    )
    parser.add_argument(
        "--labelspace_path",
        default="/home/harel/code/heracles_agents/src/heracles_agents/resources/ade20k_mit_label_space.yaml",
    )
    args = parser.parse_args()
    main(args)
