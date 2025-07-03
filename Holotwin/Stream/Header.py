#TODO: Test

import json
from .. import constants

# Protocol ver. 1

# Frame Format:
#         1.Protocol Version: 1 byte
#         2.Proto Header : 2 bytes
#         3.JSON Header : {Proto Header} bytes
#         4.Payload (Mesh/PointCloud):
#                 Mesh:
#                     4.1. Vertex Coordinates : 3xFloat32 per vertex
#                     4.2. Vertex Normals : 3xFloat32 per vertex
#                     4.3. Face Indices: 3xInt32 per face
#                 Point Cloud:
#                     4.1. Vertex Coordinates : 3xFloat32 per vertex
#                     4.2. Vertex Colors : 3xUint8 per vertex
# 
# Header Format (json):
#     1. objectType{int}:
#         0 - Point Cloud
#         1 - Mesh
#         ...
#     2. objectId{string}: Id (name) of the object. Used to reference different objects and send data to different objects.
#     3. frameId{string}: Id of the frame. Used mainly for debugging and frame drop tracking.
#     4. delay{float}: The value of the time period between frames. Used for buffering.

#     5. vertexAttributes{List}: {number of vertex attributes} * {name{string}, type{string}, size{int}, offset{int}}
#         5.1. name{string}: Used as a id of the geometry attribute.
#         5.2. type{string}: Used to specify the type of the geometry data (int32, float32, etc ...).
#         5.3. size{int}: Used to specify the size of the geometry data in terms of number of elements (not bytes).
#         5.4. offset{int}: Used to specify the byte offset of this particular geometry attribute in the frame payload.

#     6. faceAttributes{List}: {number of face attributes} * {name{string}, type{string}, size{int}, offset{int}}

#         --//--

#     7. otherAttributes{List}: {number of other attributes} * {name{string}, type{string}, size{int}, offset{int}}
    
#         --//--
    

def generateHeader(objectType = 0, objectId = "Object", frameId = "frame", delay = 0, vertexAttributeLabels = [], faceAttributeLabels = [], otherAttributeLabels = [], version = 1, returnHeaderDict = False):

    """
    Generates a header for a frame in the custom protocol format. The header contains metadata about the frame
    and is required for proper parsing and reconstruction of the frame data.

    Args:
        objectType (int, optional): The type of the object in the frame (0 for point cloud, 1 for mesh). Defaults to 0.
        objectId (str, optional): The ID (name) of the object. Used to reference different objects and send data to different objects. Defaults to "Object".
        frameId (str, optional): The ID of the frame. Used mainly for debugging and frame drop tracking. Defaults to "frame".
        delay (float, optional): The value of the time period between frames. Used for buffering. Defaults to 0.
        vertexAttributeLabels (list, optional): A list of dictionaries containing vertex attribute information. Defaults to [].
        faceAttributeLabels (list, optional): A list of dictionaries containing face attribute information. Defaults to [].
        otherAttributeLabels (list, optional): A list of dictionaries containing other attribute information. Defaults to [].
        version (int, optional): The protocol version. Defaults to 1.
        returnHeaderDict (bool, optional): Whether to return the header as a dictionary or as a formatted byte string. Defaults to False.

    Returns:
        Union[bytes, dict]: The header as a byte string (by default) or as a dictionary (if returnHeaderDict is set to True).
    """

    protoHeaderLen = 2
    headerDict = {
        "objectType": objectType,
        "objectId": objectId,
        "frameId": frameId,
        "delay": delay,
        "vertexAttributes": vertexAttributeLabels,
        "faceAttributes": faceAttributeLabels,
        "otherAttributes": otherAttributeLabels
    }

    headerJSON = json.dumps(headerDict, ensure_ascii=False).encode('utf8')
    protoHeader = len(headerJSON).to_bytes(protoHeaderLen, byteorder=constants.DEFAULT_BYTEORDER)

    if returnHeaderDict:
        return headerDict
    else:
        return protoHeader+headerJSON
