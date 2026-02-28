"""Registry system for modular, config-driven component instantiation."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A named registry that maps string keys to callable factories.

    Each module category (models, losses, datasets, etc.) maintains its own
    Registry instance. Classes register themselves via the ``register`` method
    (usable as a decorator) and are later instantiated by name from config.

    Example::

        MODELS = Registry("models")

        @MODELS.register
        class MLP(nn.Module):
            ...

        model = MODELS.build("MLP", hidden=128)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, cls: type[T]) -> type[T]:
        """Register a class under its own ``__name__``. Use as a decorator."""
        key = cls.__name__
        if key in self._registry:
            raise KeyError(
                f"{key!r} is already registered in the {self.name!r} registry"
            )
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(empty)"
            raise KeyError(
                f"{name!r} not found in the {self.name!r} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class by name, forwarding ``kwargs``."""
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list[str]:
        """Return sorted list of registered names."""
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, items={self.list()})"
