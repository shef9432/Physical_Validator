from abc import ABC, abstractmethod
import numpy as np

class BaseAdapter(ABC):
    """
    Interface for model inference adaptation.
    All custom models must implement this protocol.
    """
    @abstractmethod
    def predict(self, image: np.ndarray) -> dict:
        """
        Args:
            image: Preprocessed numpy array (BGR/RGB).
        Returns:
            A dictionary containing:
            - 'data': Model raw output (e.g., boxes or masks)
            - 'conf': Confidence score (float)
        """
        pass