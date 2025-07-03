#TODO: Test

from ..Geometry.GeometryAttribute import GeometryAttribute
from .Header import generateHeader

from dataclasses import dataclass

@dataclass
class Frame():

    """
    Frame class represents a single frame consisting of a header and vertex attributes, face attributes, and other attributes.
    """
    header: dict
    objectType: int
    objectId: str
    frameId: str
    delay: float
    version: int
    vertexAttributes: list
    faceAttributes: list
    otherAttributes: list


    @staticmethod
    def defaultHeader(vertexAttributes, faceAttributes, otherAttributes, version):

        """
        Generates a default header dictionary based on the given vertex, face, and other attributes.

        Args:
            vertexAttributes (list): A list of vertex attributes.
            faceAttributes (list): A list of face attributes.
            otherAttributes (list): A list of other attributes.
            version (int): Protocol version number.

        Returns:
            dict: A dictionary containing the default header information.
        """

        if faceAttributes:
            objectType = 1
        else:
            objectType = 0
        
        vertexAttributeLabels = []
        faceAttributeLabels = []
        otherAttributeLabels = []
        offset = 0
        for i,attribute in enumerate(vertexAttributes):
            vertexAttributeLabels.append(attribute.label)
            attribute.setOffset(offset)
            offset += len(attribute.buf)

        for i,attribute in enumerate(faceAttributes):
            faceAttributeLabels.append(attribute.label)
            attribute.setOffset(offset)
            offset += len(attribute.buf)

        for i,attribute in enumerate(otherAttributes):
            otherAttributeLabels.append(attribute.label)
            attribute.setOffset(offset)
            offset += len(attribute.buf)

        headerDict = generateHeader(objectType=objectType, 
                            vertexAttributeLabels=vertexAttributeLabels, 
                            faceAttributeLabels=faceAttributeLabels, 
                            otherAttributeLabels=otherAttributeLabels,
                            version = version, 
                            returnHeaderDict=True)

        return headerDict

    def __str__(self):
        """
        Returns the string representation of the Frame object, displaying the header information.
        """
        return f"{self.header}"

    def __init__(self, header = {}, vertexAttributes = [], faceAttributes = [], otherAttributes = [], version = 1):

        """
        Initializes the Frame object with the provided header and attributes.
        If the header is not provided, it will generate a default header using the attributes.

        Args:
            header (dict): A dictionary containing header information. Defaults to an empty dictionary.
            vertexAttributes (list): A list of vertex attributes. Defaults to an empty list.
            faceAttributes (list): A list of face attributes. Defaults to an empty list.
            otherAttributes (list): A list of other attributes. Defaults to an empty list.
            version (int): Frame version number. Defaults to 1.
        """

        self.vertexAttributes = vertexAttributes
        self.faceAttributes = faceAttributes
        self.otherAttributes = otherAttributes
        self.version = version
        
        if header:
            self.header = header
            self.objectType = header["objectType"]
            self.objectId = header["objectId"]
            self.frameId = header["frameId"]
            self.delay = header["delay"]
        else:
            self.header = self.defaultHeader(vertexAttributes, faceAttributes, otherAttributes, self.version)
            self.objectType = header["objectType"]
            self.objectId = header["objectId"]
            self.frameId = header["frameId"]
            self.delay = header["delay"]


