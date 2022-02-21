""" utility functions for manipulating MJCF XML models. """
""" from https://github.com/StanfordVL/robosuite/blob/master/robosuite/utils/mjcf_utils.py """

import xml.etree.ElementTree as ET
import os
import numpy as np

from .models import assets_root


RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]

MUJOCO_NAMED_ATTRIBUTES = {
    "class",
    "childclass",
    "name",
    "objname",
    "material",
    "texture",
    "joint",
    "joint1",
    "joint2",
    "jointinparent",
    "geom",
    "geom1",
    "geom2",
    "mesh",
    "fixed",
    "actuator",
    "objname",
    "tendon",
    "tendon1",
    "tendon2",
    "slidesite",
    "cranksite",
    "body",
    "body1",
    "body2",
    "hfield",
    "target",
    "prefix",
    "site",
}

SENSOR_TYPES = {
    "touch",
    "accelerometer",
    "velocimeter",
    "gyro",
    "force",
    "torque",
    "magnetometer",
    "rangefinder",
    "jointpos",
    "jointvel",
    "tendonpos",
    "tendonvel",
    "actuatorpos",
    "actuatorvel",
    "actuatorfrc",
    "ballangvel",
    "jointlimitpos",
    "jointlimitvel",
    "jointlimitfrc",
    "tendonlimitpos",
    "tendonlimitvel",
    "tendonlimitfrc",
    "framepos",
    "framequat",
    "framexaxis",
    "frameyaxis",
    "framezaxis",
    "framelinvel",
    "frameangvel",
    "framelinacc",
    "frameangacc",
    "subtreecom",
    "subtreelinvel",
    "subtreeangmom",
    "user",
}


def xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(assets_root, xml_path)
    return full_path


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])


def set_alpha(node, alpha=0.1):
    """
    Sets all a(lpha) field of the rgba attribute to be @alpha
    for @node and all subnodes
    used for managing display
    """
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rgba"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_joint(**kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.
    """

    element = ET.Element("joint", attrib=kwargs)
    return element


def new_actuator(joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        joint: type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator type. Defaults to "actuator"

    """
    element = ET.Element(act_type, attrib=kwargs)
    element.set("joint", joint)
    return element


def new_site(name, rgba=RED, pos=(0, 0, 0), size=(0.005,), **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        name (str): site name.
        rgba: color and transparency. Defaults to solid red.
        pos: 3d position of the site.
        size ([float]): site size (sites are spherical by default).
    """
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["pos"] = array_to_string(pos)
    kwargs["size"] = array_to_string(size)
    kwargs["name"] = name
    element = ET.Element("site", attrib=kwargs)
    return element


def new_geom(geom_type, size, pos=(0, 0, 0), rgba=RED, group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    Args:
        geom_type (str): type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size: geom size parameters.
        pos: 3d position of the geom frame.
        rgba: color and transparency. Defaults to solid red.
        group: the integrer group that the geom belongs to. useful for
            separating visual and physical elements.
    """
    kwargs["type"] = str(geom_type)
    kwargs["size"] = array_to_string(size)
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["group"] = str(group)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("geom", attrib=kwargs)
    return element


def new_body(name=None, pos=None, **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): body name.
        pos: 3d position of the body frame.
    """
    if name is not None:
        kwargs["name"] = name
    if pos is not None:
        kwargs["pos"] = array_to_string(pos)
    element = ET.Element("body", attrib=kwargs)
    return element


def new_inertial(name=None, pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        mass: The mass of inertial
    """
    if mass is not None:
        kwargs["mass"] = str(mass)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("inertial", attrib=kwargs)
    return element

# add prefix from robosuite
def add_prefix(
    root,
    prefix,
    tags="default",
    attribs="default",
    exclude=None,
):
    """
    Find all element(s) matching the requested @tag, and appends @prefix to all @attributes if they exist.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        prefix (str): Prefix to add to all specified attributes
        tags (str or list of str or set): Tag(s) to search for in this ElementTree. "Default" corresponds to all tags
        attribs (str or list of str or set): Element attribute(s) to append prefix to. "Default" corresponds
            to all attributes that reference names
        exclude (None or function): Filtering function that should take in an ET.Element or a string (attribute) and
            return True if we should exclude the given element / attribute from having any prefixes added
    """
    # Standardize tags and attributes to be a set
    if tags != "default":
        tags = {tags} if type(tags) is str else set(tags)
    if attribs == "default":
        attribs = MUJOCO_NAMED_ATTRIBUTES
    attribs = {attribs} if type(attribs) is str else set(attribs)

    # Check the current element for matching conditions
    if (tags == "default" or root.tag in tags) and (exclude is None or not exclude(root)):
        for attrib in attribs:
            v = root.get(attrib, None)
            # Only add prefix if the attribute exist, the current attribute doesn't already begin with prefix,
            # and the @exclude filter is either None or returns False
            if v is not None and not v.startswith(prefix) and (exclude is None or not exclude(v)):
                root.set(attrib, prefix + v)
    # Continue recursively searching through the element tree
    for r in root:
        add_prefix(root=r, prefix=prefix, tags=tags, attribs=attribs, exclude=exclude)


def _element_filter(element, parent):
    """
    Default element filter to be used in sort_elements. This will filter for the following groups:

        :`'root_body'`: Top-level body element
        :`'bodies'`: Any body elements
        :`'joints'`: Any joint elements
        :`'actuators'`: Any actuator elements
        :`'sites'`: Any site elements
        :`'sensors'`: Any sensor elements
        :`'contact_geoms'`: Any geoms used for collision (as specified by group 0 (default group) geoms)
        :`'visual_geoms'`: Any geoms used for visual rendering (as specified by group 1 geoms)

    Args:
        element (ET.Element): Current XML element that we are filtering
        parent (ET.Element): Parent XML element for the current element

    Returns:
        str or None: Assigned filter key for this element. None if no matching filter is found.
    """
    # Check for actuator first since this is dependent on the parent element
    if parent is not None and parent.tag == "actuator":
        return "actuators"
    elif element.tag == "joint":
        # Make sure this is not a tendon (this should not have a "joint", "joint1", or "joint2" attribute specified)
        if element.get("joint") is None and element.get("joint1") is None:
            return "joints"
    elif element.tag == "body":
        # If the parent of this does not have a tag "body", then this is the top-level body element
        if parent is None or parent.tag != "body":
            return "root_body"
        return "bodies"
    elif element.tag == "site":
        return "sites"
    elif element.tag in SENSOR_TYPES:
        return "sensors"
    elif element.tag == "geom":
        # Only get collision and visual geoms (group 0 / None, or 1, respectively)
        group = element.get("group")
        if group in {None, "0", "1"}:
            return "visual_geoms" if group == "1" else "contact_geoms"
    else:
        # If no condition met, return None
        return None


def sort_elements(root, parent=None, element_filter=None, _elements_dict=None):
    """
    Utility method to iteratively sort all elements based on @tags. This XML ElementTree will be parsed such that
    all elements with the same key as returned by @element_filter will be grouped as a list entry in the returned
    dictionary.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through
        parent (ET.Element): Parent of the root node. Default is None (no parent node initially)
        element_filter (None or function): Function used to filter the incoming elements. Should take in two
            ET.Elements (current_element, parent_element) and return a string filter_key if the element
            should be added to the list of values sorted by filter_key, and return None if no value should be added.
            If no element_filter is specified, defaults to self._element_filter.
        _elements_dict (dict): Dictionary that gets passed to recursive calls. Should not be modified externally by
            top-level call.

    Returns:
        dict: Filtered key-specific lists of the corresponding elements
    """
    # Initialize dictionary and element filter if None is set
    if _elements_dict is None:
        _elements_dict = {}
    if element_filter is None:
        element_filter = _element_filter

    # Parse this element
    key = element_filter(root, parent)
    if key is not None:
        # Initialize new entry in the dict if this is the first time encountering this value, otherwise append
        if key not in _elements_dict:
            _elements_dict[key] = [root]
        else:
            _elements_dict[key].append(root)

    # Loop through all possible subtrees for this XML recurisvely
    for r in root:
        _elements_dict = sort_elements(
            root=r, parent=root, element_filter=element_filter, _elements_dict=_elements_dict
        )

    return _elements_dict