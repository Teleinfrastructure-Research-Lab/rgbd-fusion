#TODO: Test

import numpy as np
import sys
import struct
from .. import constants

class GeometryAttribute:

    """
    GeometryAttribute class represents an attribute of a geometry, such as vertex or face attributes.
    It is capable of serializing and deserializing different data types, including numpy arrays, integers,
    floats, strings, and bytes.
    """

    def getByteLength(self, size, dtype_str):

        """
        Calculates the byte length of the given data type and size.

        Args:
            size (int): The size of the data.
            dtype_str (str): The data type as a string.
            intByteLength (int): The byte length of integers. Defaults to constants.INT_BYTE_LENGTH.
            floatByteLength (int): The byte length of floats. Defaults to constants.FLOAT_BYTE_LENGTH.

        Returns:
            int: The byte length of the data type and size.
        """
        
        if size > 1:
            if dtype_str == "string" or dtype_str == "bytes":
                return size
            else:
                dtype = np.dtype(dtype_str)
                return size * dtype.itemsize
        else:
            if dtype_str == "int":
                return self.intByteLength
            elif dtype_str == "float":
                return self.floatByteLength
            else:
                return 0       

    def setOffset(self, offset):

        """
        Sets the offset for the attribute and updates the label.

        Args:
            offset (int): The offset value.
        """

        self.offset = offset
        self.label = self.generateLabel()

    def generateLabel(self):

        """
        Generates a label containing the name, data type, size, and offset of the attribute.

        Returns:
            dict: The label dictionary.
        """

        label = {
            "name": self.name,
            "dataType": self.dataType,
            "size": self.length,
            "offset": self.offset
        }

        return label

    def Serialize(self):

        """
        Serializes the attribute data according to its data type.

        Returns:
            bytes: The serialized attribute data.
        """

        #TODO: Add serialization for torch tensors
        #TODO: Add compression

        # Handle serialization for numpy arrays
        if type(self.data) == np.ndarray:
            self.dataType = str(self.data.dtype)
            self.length = self.data.size
            if self.data.dtype.byteorder != '|':
                if self.byteOrder == 'little':
                    if self.data.dtype.byteorder == '=':
                        if sys.byteorder == 'little':
                            buf = self.data.tobytes()
                        else:
                            buf = self.data.byteswap(True).tobytes()
                    elif self.data.dtype.byteorder == '<':
                        buf = self.data.tobytes()
                    else:
                        buf = self.data.byteswap(True).tobytes()
                else:
                    if self.data.dtype.byteorder == '=':
                        if sys.byteorder == 'little':
                            buf = self.data.byteswap(True).tobytes()
                        else:
                            buf = self.data.tobytes()
                    elif self.data.dtype.byteorder == '<':
                        buf = self.data.byteswap(True).tobytes()
                    else:
                        buf = self.data.tobytes()      
            else:
                buf = self.data.tobytes()
    
        # Handle serialization for integers
        elif type(self.data) == int:
            self.dataType = self.defaultInt
            self.length = 1
            buf = self.data.to_bytes(self.intByteLength, self.byteOrder)

        # Handle serialization for floats
        elif type(self.data) == float:
            self.dataType = self.defaultFloat
            self.length = 1
            if self.byteOrder == 'little':
                o = '<'
            else:
                o = '>'

            if self.floatByteLength == 4:
                s = 'f'
            elif self.floatByteLength == 8:
                s = 'd'
            buf = struct.pack(o+s, self.data)

        # Handle serialization for strings
        elif type(self.data) == str:
            self.dataType = 'string'
            self.length = len(self.data)
            buf = self.data.encode('utf-8')

        # Handle serialization for bytes
        elif type(self.data) == bytes:
            self.dataType = 'byte'
            self.length = len(self.data)
            buf = self.data
        
        else:
            self.dataType = str(type(self.data))
            print(f"----GeometryAttribute: Data Type {type(self.data)} is not supported for serialization")
            buf = b""

        return buf

    def Deserialize(self):

        """
        Deserializes the attribute data from the buffer according to its data type.

        Returns:
            The deserialized attribute data.
        """

        #TODO: Add serialization for torch tensors
        #TODO: Add decompression

        # Handle deserialization for multi-element data (byte strings, charecter strings and numpy arrays)
        if self.length > 1:
            if self.dataType == "string":
                data = self.buf.decode('utf-8')
            elif self.dataType == "byte":
                data = self.buf
            else:
                dataType = np.dtype(self.dataType)
                if self.byteOrder == "little":
                    dataType.newbyteorder("<")
                elif self.byteOrder == "big":
                    dataType.newbyteorder(">")
                
                data = np.frombuffer(self.buf, dtype=dataType)

        # Handle deserialization for single-element data (integers and floats)    
        else:
            if self.dataType == 'int':
                data = int.from_bytes(self.buf, self.byteOrder)

            elif self.dataType == 'float':
                if self.byteOrder == 'little':
                    o = '<'
                else:
                    o = '>'
                if self.floatByteLength == 4:
                    s = 'f'
                elif self.floatByteLength == 8:
                    s = 'd'
                data = struct.unpack(o+s, self.buf)[0]
        
        return data

    def __str__(self):
        """
        Returns a string representation of the attribute's label.

        Returns:
            str: The string representation of the label.
        """
        return f"{self.label}"

    def __init__(self, name = None, data = None, buf = None, label = None):

        """
        Initializes the GeometryAttribute object with the provided name, data, buffer, and label.
        If name and data are provided, the object will be initialized using name and data.
        If buffer and label are provided, the object will be initialized using buffer and label.

        Args:
            name (str): The name of the attribute. Defaults to None.
            data: The data associated with the attribute. Defaults to None.
            buf (bytes): The serialized buffer of the attribute. Defaults to None.
            label (dict): The label containing metadata about the attribute. Defaults to None.
        """

        self.defaultInt = constants.DEFAULT_INT 
        self.defaultFloat = constants.DEFAULT_FLOAT
        self.intByteLength = constants.INT_BYTE_LENGTH
        self.floatByteLength = constants.FLOAT_BYTE_LENGTH
        self.byteOrder = constants.DEFAULT_BYTEORDER
        self.offset = 0

        if (name is not None and data is not None) and (buf is None and label is None):
            self.name = name
            self.data = data
            self.buf = self.Serialize()
            self.label = self.generateLabel()
        elif (name is None and data is None) and (buf is not None and label is not None):
            self.label = label
            self.name = label["name"]
            self.length = label["size"]
            self.dataType = label["dataType"]
            self.offset = label["offset"]
            self.byteLength = self.getByteLength(self.length, self.dataType)
            self.buf = buf[self.offset:self.offset+self.byteLength]            
            self.data = self.Deserialize()
        else:
            self.name = name
            self.data = data
            self.buf = buf
            self.label = label
            print("----GeometryAttribute: Insuffficient data provided to constructor. Please provide either name and data or buffer and label.")
