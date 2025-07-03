#TODO: Try using __getattr__ / __setattr__ isntead of get()/set() ?
#TODO: TEST!

import threading

class TSafeValue:

    """
    TSafeValue is a class for safely accessing and modifying a value in a multi-threaded environment.
    """

    def set(self, value):
        """
        Sets the value in a thread-safe manner.
        Args:
            value: The new value to be set.
        """
        with self.lock:
            self.value = value

    def get(self):
        """
        Gets the current value in a thread-safe manner.
        Returns:
            The current value.
        """
        with self.lock:
            return self.value

    def __init__(self,value):
        """
        Initializes the TSafeValue instance with the specified initial value.
        Args:
            value: The initial value.
        """
        self.lock = threading.Lock()
        self.value = value

class TSafeSet:

    """
    TSafeSet is a class for safely accessing and modifying a set in a multi-threaded environment.
    """

    def __init__(self):
        """
        Initializes the TSafeSet instance.
        """
        self.set = set()
        self.lock = threading.Lock()

    def add(self, item):
        """
        Adds an item to the set in a thread-safe manner.
        Args:
            item: The item to be added.
        """
        with self.lock:
            self.set.add(item)

    def remove(self, item):
        """
        Removes an item from the set in a thread-safe manner.
        Args:
            item: The item to be removed.
        """
        with self.lock:
            self.set.remove(item)

    def get(self):
        """
        Gets the current set in a thread-safe manner.
        Returns:
            The current set.
        """
        with self.lock:
            return self.set

    def __contains__(self, item):
        """
        Checks if the item is in the set in a thread-safe manner.
        Args:
            item: The item to be checked.
        Returns:
            True if the item is in the set, otherwise False.
        """
        with self.lock:
            return item in self.set

    def __len__(self):
        """
        Gets the length of the set in a thread-safe manner.
        Returns:
            The length of the set.
        """
        with self.lock:
            return len(self.set)

    def __iter__(self):
        """
        Creates an iterator of the set in a thread-safe manner.
        Returns:
            An iterator of the set.
        """
        with self.lock:
            return iter(self.set.copy())