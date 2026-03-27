from abc import ABC, abstractmethod

class BaseSequenceProcessor(ABC):
    
    @abstractmethod
    def config(self, **kwargs):
        """ Configure the sequence processor with given parameters."""
        pass

    @property
    def name(self):
        """ Return the name of the sequence processor."""
        return self.__class__.__name__