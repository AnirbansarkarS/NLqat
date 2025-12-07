from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    """
    
    @abstractmethod
    def load(self, source: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Loads data from a source and returns a list of dictionaries (documents).
        """
        pass
