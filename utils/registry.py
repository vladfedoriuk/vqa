"""Registry."""
from collections.abc import Callable, MutableMapping
from enum import Enum
from typing import ClassVar, Generic, TypeVar

ValueType = TypeVar("ValueType")


class RegistryKey(Enum):
    """
    Registry key.

    The case-specific registry keys are defined as subclasses.
    """

    pass


KeyType = TypeVar("KeyType", bound=RegistryKey)

OtherType = TypeVar("OtherType")


class Registry(Generic[KeyType, ValueType]):
    """
    Registry.

    It can be used to register and get values.
    The case-specific registries are defined as subclasses.

    The registry is a singleton.
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    _REGISTRY_TYPE = MutableMapping[KeyType, ValueType]
    internal_registry: ClassVar[_REGISTRY_TYPE] = {}

    def register(
        self, key: RegistryKey, value: ValueType | None = None
    ) -> (ValueType | Callable[[ValueType], ValueType]):
        """
        Register a value in the registry.

        :param key: The key to register the value for.
        :param value: The value to register.
        """
        # noqa: D202
        def register_value(_value: ValueType) -> ValueType:
            self.internal_registry[key] = _value
            return _value

        return register_value if value is None else register_value(value)

    def get(self, key: RegistryKey) -> ValueType | None:
        """
        Get a value from the registry.

        :param key: The key to get the value for.
        :return: The value or None if the key is not in the registry.
        """
        return self.get_or(key, None)

    def get_or(self, key: RegistryKey, default: OtherType) -> ValueType | OtherType:
        """
        Get a value from the registry or a default value.

        :param key: The key to get the value for.
        :param default: The default value to return if the key is not in the registry.
        :return: The value or the default value if the key is not in the registry.
        """
        return self.internal_registry.get(key, default)
