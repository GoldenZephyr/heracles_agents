# Conventions (as of 09-11-2025)
# We had been testing with a prompt that excluded 2D place nodes entirely & directly connected regions to objects
# The entry function for that is unchanged: `scene_graph_to_prompt`
#
# We want to test with a prompt that accurately reflects the hierarchy of the ontology
# The entry function for that is: `scene_graph_to_prompt_full`

import spark_dsg


class PromptingFailure(Exception):
    pass


""" Methods for encoding a scene graph with ONLY Objects and Regions/Rooms """


def get_region_parents_of_object(object_node, scene_graph):
    """Method to return the string of the object node's grandparent region; this requires traversing the intermediate place layer"""
    # Get the parents; returns a set of gtsam ids
    parent_place_gtsam_ids = object_node.parents()
    if not parent_place_gtsam_ids:
        return "none"
    # Get the parent regions
    parent_region_node_ids = set()
    for parent_place_gtsam_id in parent_place_gtsam_ids:
        parent_place_node = scene_graph.get_node(parent_place_gtsam_id)
        parent_region_gtsam_ids = parent_place_node.parents()
        if not parent_region_gtsam_ids:
            continue
        for parent_region_gtsam_id in parent_region_gtsam_ids:
            parent_region_node = scene_graph.get_node(parent_region_gtsam_id)
            parent_region_node_ids.add(parent_region_node.id.str(True))
    if not parent_region_node_ids:
        return "none"
    return parent_region_node_ids


def region_to_string(region_node, scene_graph):
    attrs = region_node.attributes
    symbol = region_node.id.str(True)
    region_labelspace = scene_graph.get_labelspace(4, 0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category(attrs.semantic_label)
    region_string = f"\n-\t(id={symbol}, type={semantic_type})"
    return region_string


def object_to_string_region_parent(object_node, scene_graph):
    """Method to return the stirng of an object node; excludes the parent place and directly encodes the grandparent region"""
    attrs = object_node.attributes
    symbol = object_node.id.str(True)
    object_labelspace = scene_graph.get_labelspace(2, 0)
    if not object_labelspace:
        raise PromptingFailure("No available object labelspace")
    semantic_type = object_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]:.2f},{attrs.position[1]:.2f})"
    parent_regions = get_region_parents_of_object(object_node, scene_graph)
    object_string = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_regions={parent_regions})"
    return object_string


def scene_graph_to_prompt(scene_graph):
    """Method to produce a text encoding of a spark_dsg DynamicSceneGraph. This excludes 2D places and directly encodes a connection from objects to regions"""
    # Add the objects
    objects_string = ""
    for object_node in scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        objects_string += object_to_string_region_parent(
            object_node, scene_graph
        )  # Use the method that skips the place parent node
    # Add the regions
    regions_string = ""
    for region_node in scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        regions_string += region_to_string(region_node, scene_graph)
    # Compose the strings into a single text encoding
    scene_graph_string = f"<Scene Graph>\nObjects: {objects_string}\nRegions: {regions_string}</Scene Graph>"
    return scene_graph_string


""" Methods for encoding a scene graph with Objects, Places, and Regions/Rooms """


def region_to_string_full(region_node, scene_graph):
    """Method to compose a string encoding of a region: unique id, semantic type, and position"""
    attrs = region_node.attributes
    symbol = region_node.id.str(True)
    region_labelspace = scene_graph.get_labelspace(4, 0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]:.2f}, {attrs.position[1]:.2f})"
    region_string = f"\n-\t(id={symbol}, type={semantic_type}, pos={position})"
    return region_string


def place_to_string_full(place_node, scene_graph):
    """Method to compose a string encoding of a mesh place: unique id, sibling place unique ids, parent region/room unique ids"""
    symbol = place_node.id.str(True)
    # Get the siblings
    sibling_node_ids = set()
    sibling_gtsam_ids = place_node.siblings()
    for sibling_gtsam_id in sibling_gtsam_ids:
        sibling_node = scene_graph.get_node(sibling_gtsam_id)
        sibling_node_ids.add(sibling_node.id.str(True))
    if not sibling_node_ids:
        sibling_node_ids = "none"
    # Get the parents
    parent_region_node_ids = set()
    parent_region_gtsam_ids = place_node.parents()
    for parent_region_gtsam_id in parent_region_gtsam_ids:
        parent_region_node = scene_graph.get_node(parent_region_gtsam_id)
        parent_region_node_ids.add(parent_region_node.id.str(True))
    if not parent_region_node_ids:
        parent_region_node_ids = "none"
    # Compose the string
    place_string = f"\n-\t(id={symbol}, siblings={sibling_node_ids}, parent_regions={parent_region_node_ids})"
    return place_string


def object_to_string_full(object_node, scene_graph):
    """Method to compose a string encoding an object: unique id, semantic type, position, bounding box pos/dim, parent place unique ids"""
    attrs = object_node.attributes
    symbol = object_node.id.str(True)
    object_labelspace = scene_graph.get_labelspace(2, 0)
    if not object_labelspace:
        raise PromptingFailure("No available object labelspace")
    semantic_type = object_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]:.2f},{attrs.position[1]:.2f})"
    parent_place_node_ids = set()
    parent_place_gtsam_ids = object_node.parents()
    for parent_place_gtsam_id in parent_place_gtsam_ids:
        parent_place_node = scene_graph.get_node(parent_place_gtsam_id)
        parent_place_node_ids.add(parent_place_node.id.str(True))
    if not parent_place_node_ids:
        parent_place_node_ids = "none"
    # Compose the string
    object_string = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_places={parent_place_node_ids})"
    return object_string


def scene_graph_to_prompt_full(scene_graph, place_layer_name):
    """Method to produce a text encoding of a spark_dsg DynamicSceneGraph. This includes 2D places."""
    # Add the objects: unique id, semantic label, position, bounding box, parent 2D place uid
    objects_string = ""
    for object_node in scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        objects_string += object_to_string_full(object_node, scene_graph)
    # Add the places: unique id, sibling ids, parent region unique id
    places_string = ""
    for place_node in scene_graph.get_layer(place_layer_name).nodes:
        places_string += place_to_string_full(place_node, scene_graph)
    # Add the regions: unique id, semantic label, position
    regions_string = ""
    for region_node in scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        regions_string += region_to_string_full(region_node, scene_graph)
    scene_graph_string = f"<Scene Graph>\nObjects: {objects_string}\nPlaces: {places_string}\n Regions: {regions_string}\n</Scene Graph>"
    return scene_graph_string
