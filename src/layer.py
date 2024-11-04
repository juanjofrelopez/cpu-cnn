from abc import ABC, abstractmethod
from numpy import ndarray


class Layer(ABC):
    @abstractmethod
    def forward() -> ndarray:
        pass

    @abstractmethod
    def backward() -> ndarray:
        pass
