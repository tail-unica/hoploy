from abc import ABC, abstractmethod

class BaseLogitsProcessor(ABC):
    
    @abstractmethod
    def config(self, **kwargs):
        """ Configure the logits processor with given parameters."""
        pass

    @property
    def name(self):
        """ Return the name of the logits processor."""
        return self.__class__.__name__