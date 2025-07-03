#TODO: TEST!

from .. import constants

class FileStreamer():

    """
    FileStreamer class writes binary data to a file with a specified path.
    The class is responsible for writing frames of data to the file.
    """

    def start(self):

        """
        Opens the binary file for writing and initializes the frame counter.
        Sets the started attribute to True, indicating that the FileStreamer has started.
        """
        
        if self.started == False:
            self.fp = open(self.path, "wb")
            self.count = 0
            self.started = True
            print(f"----FileStreamer: Starting.")

    def stop(self):

        """
        Closes the binary file, stops the FileStreamer, and sets the started attribute to False.
        """

        if self.started == True:
            self.fp.close()
            self.started = False
            print(f"----FileStreamer: Closing File.")

    def sendFrame(self, fl, fd):

        """
        Writes a frame to the binary file.
        The frame is a combination of frame length (fl) and frame data (fd).
        Stops the FileStreamer if the maximum frame count (maxCount) has been reached.
        """

        buf = fl + fd
        if self.count <= self.maxCount:
            try:
                self.fp.write(buf)
            except:
                print("----FileStreamer: Unable to write to file!")
            self.count += 1
        else:
            self.stop()

    def __init__(self, path):

        """
        Initializes the FileStreamer object with the specified file path.
        Sets initial values for file pointer, frame count, maximum frame count, started and enabled attributes.
        """

        self.path = path
        self.fp = None
        self.count = 0
        self.maxCount = constants.FILESTREAMER_MAX_COUNT #This variable controls the number of frames captured
        self.started = False
        self.enabled = False


# ==========================================================
# Test
# ==========================================================
if __name__ == "__main__":
    import numpy as np
    import time
    import math
    import secrets
    gs = FileStreamer("fcap.txt")
    gs.start()


    points_num = 512*424
    def random_bytes(length):
        return bytearray(secrets.token_bytes(length))

    
    fps =15
    while(1):
        ts = time.time()

        gs.sendFrame(random_bytes(4),random_bytes((4)+(points_num*12)+(points_num*3)))
        if gs.started == False:
            break

        te = time.time()
        while (te-ts) < (1/fps):
            te = time.time()
