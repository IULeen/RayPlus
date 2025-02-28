from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from ray import cloudpickle
from utils.serialization import pickle_dumps


class ActorlessConfig:

    def __init__(
        self,
        serialized_deployment_def: bytes,
        needs_pickle: bool = True,
    ):
        # Store serialized versions of code properties.
        self.serialized_deployment_def = serialized_deployment_def

        # Deserialize properties when first accessed. See @property methods.
        self._deployment_def = None

        self.needs_pickle = needs_pickle


    @classmethod
    def create(
        cls,
        _func_or_class: Union[Callable, str],
    ):
        """Create a ReplicaConfig from deserialized parameters."""
        deployment_def = _func_or_class

        if not callable(deployment_def) and not isinstance(deployment_def, str):
            raise TypeError("@actorless_decorator.actorless must be called on a class or function.")

        if not isinstance(deployment_def, (Callable, str)):
            raise TypeError(
                f'Got invalid type "{type(deployment_def)}" for '
                "deployment_def. Expected deployment_def to be a "
                "class, function, or string."
            )

        config = cls(
            pickle_dumps(
                deployment_def,
                f"Could not serialize the deployment {repr(deployment_def)}",
            ),
        )

        config._deployment_def = deployment_def

        return config
    

    @property
    def deployment_def(self) -> Union[Callable, str]:
        """The code, or a reference to the code, that this replica runs.

        For Python replicas, this can be one of the following:
            - Function (Callable)
            - Class (Callable)
            - Import path (str)

        For Java replicas, this can be one of the following:
            - Class path (str)
        """
        if self._deployment_def is None:
            if self.needs_pickle:
                self._deployment_def = cloudpickle.loads(self.serialized_deployment_def)
            else:
                self._deployment_def = self.serialized_deployment_def.decode(
                    encoding="utf-8"
                )

        return self._deployment_def
    

    # TODO: need to check before using this function
    @property
    def init_args(self) -> Optional[Union[Tuple[Any], bytes]]:
        """The init_args for a Python class.

        This property is only meaningful if deployment_def is a Python class.
        Otherwise, it is None.
        """
        if self._init_args is None:
            if self.needs_pickle:
                self._init_args = cloudpickle.loads(self.serialized_init_args)
            else:
                self._init_args = self.serialized_init_args

        return self._init_args