from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    @abstractmethod
    def recommend(self, inputs):
        """ Generate output based on the given inputs."""
        pass
    
    @abstractmethod
    def config(self, **kwargs):
        """ Configure the model with given parameters. This method can be used to set up the model before generation."""
        pass

    @abstractmethod
    def distill(self, **kwargs):
        """
        From API request schema to model input.
        """
        pass

    @abstractmethod
    def encode(self, value, **kwargs):
        """ Encode the distilled values. This method can be used to convert the distilled values into a format suitable for generation."""
        pass

    @abstractmethod
    def decode(self, value):
        """ Decode the generated values. This method can be used to convert the generated values into a format suitable for further processing."""
        pass

    @abstractmethod
    def expand(self, values) -> dict:
        """
        From model output to API response schema.
        """
        pass
