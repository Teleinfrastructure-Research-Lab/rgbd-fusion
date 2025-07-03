#TODO: TEST!

import struct
import json

from .. import constants
from ..Geometry.GeometryAttribute import GeometryAttribute
from .Frame import Frame


class FileReader():

    """
    FileReader class reads frame data from a file with a specified path.
    """

    def readFrame(self):

        """
        Reads and returns a frame from the binary file.
        The frame consists of a header, vertex attributes, face attributes, and other attributes.
        """ 

        # Read frame length (fl) based on byte order
        if self.byteorder == 'little':
            fl = struct.unpack('<I', self.readBytes(4))[0]
        else:
            fl = struct.unpack('>I', self.readBytes(4))[0]
        # Read protocol version (ver) based on byte order        
        if self.byteorder == 'little':
            ver = struct.unpack('b', self.readBytes(1))[0]
        else:
            ver = struct.unpack('b', self.readBytes(1))[0]
        # Read header length (protoehader) based on byte order
        if self.byteorder == 'little':
            protoheader = struct.unpack('<H', self.readBytes(2))[0]
        else:
            protoheader = struct.unpack('>H', self.readBytes(2))[0]
        
        # Decode header and load it as a JSON object
        header = self.readBytes(protoheader).decode()
        header = json.loads(header)

        # Initialize attribute lists
        vertexAttributes = []
        faceAttributes = []
        otherAttributes = []

        # Read buffer (buf) for the remaining data in the frame
        buf = self.readBytes(fl - (7+protoheader))

        # Populate attribute lists with GeometryAttribute objects
        for attr in header["vertexAttributes"]:
            vertexAttributes.append(GeometryAttribute(buf = buf, label = attr))
        for attr in header["faceAttributes"]:
            faceAttributes.append(GeometryAttribute(buf = buf, label = attr))
        for attr in header["otherAttributes"]:
            otherAttributes.append(GeometryAttribute(buf = buf, label = attr))
        
        # Move the file cursor to the next frame and update the cursor position
        self.fp.seek(self.cursor+fl)
        self.cursor += fl

        return Frame(header, vertexAttributes, faceAttributes, otherAttributes)

    def readBytes(self, length):

        """
        Reads and returns a specified number of bytes (length) from the current file position.
        Raises StopIteration if end-of-file (EOF) is reached before reading the specified length.
        """

        cur = self.fp.tell()
        data = self.fp.read(length)
        if self.fp.tell() < cur + length:
            self.eof = True
            print("----FileReader: EOF!")
            raise StopIteration
            return bytearray(length)
        return data
    
    def reset(self):

        """
        Resets the file cursor to the beginning of the file and sets eof attribute to False.
        """

        self.cursor = 0
        self.fp.seek(0)
        self.eof = False
    
    def readOffsetBytes(self, offset, length):

        """
        Reads and returns a specified number of bytes (length) from the given offset position in the file.
        Raises StopIteration if end-of-file (EOF) is reached before reading the specified length.
        """

        cur = self.fp.tell()
        self.fp.seek(offset)
        data = self.fp.read(length)
        if self.fp.tell() < cur + length:
            self.eof = True
            print("----FileReader: EOF!")
            self.fp.seek(cur)
            raise StopIteration
            return bytearray(length)
        self.fp.seek(cur)
        
        return data

    def __next__(self):

        """
        Reads and returns the next frame in the file. For use with "for-in" loop syntax.
        """

        frame = self.readFrame()
        return frame

    def __iter__(self):

        """
        Resets the file cursor to the beginning of the file and returns the FileReader object as an iterator. For use with "for-in" loop syntax.
        """

        self.reset()
        return self

    def __enter__(self):

        """
        Opens the binary file for reading and returns the FileReader object when used in a context manager.
        """

        self.fp = open(self.path, "rb")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        """
        Closes the binary file when exiting the context manager.
        """
        
        self.fp.close()

    def __init__(self, path):

        """
        Initializes the FileReader object with the specified file path, sets default byte order, integer and float byte
        lengths, and initializes the file pointer, cursor position, and end-of-file (EOF) attribute.
        """
        
        self.byteorder = constants.DEFAULT_BYTEORDER
        self.intByteLength = constants.INT_BYTE_LENGTH 
        self.floatByteLength = constants.FLOAT_BYTE_LENGTH
        self.path = path
        self.fp = None
        self.cursor = 0
        self.eof = False


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    with FileReader("sceneCap.txt") as fr:
        for frame in fr:
            print(frame)