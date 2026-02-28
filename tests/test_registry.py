"""Tests for the Registry system."""

import pytest

from simpleml.registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register
        class Foo:
            pass

        assert reg.get("Foo") is Foo

    def test_build_instantiates_with_kwargs(self):
        reg = Registry("test")

        @reg.register
        class Bar:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        obj = reg.build("Bar", x=1, y=2)
        assert obj.x == 1
        assert obj.y == 2

    def test_duplicate_register_raises(self):
        reg = Registry("test")

        @reg.register
        class Dup:
            pass

        with pytest.raises(KeyError, match="already registered"):
            reg.register(Dup)

    def test_get_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("NonExistent")

    def test_list_returns_sorted_names(self):
        reg = Registry("test")

        @reg.register
        class Bravo:
            pass

        @reg.register
        class Alpha:
            pass

        assert reg.list() == ["Alpha", "Bravo"]

    def test_contains(self):
        reg = Registry("test")

        @reg.register
        class Item:
            pass

        assert "Item" in reg
        assert "Missing" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0

        @reg.register
        class A:
            pass

        assert len(reg) == 1

    def test_repr(self):
        reg = Registry("mymodules")
        assert "mymodules" in repr(reg)

    def test_register_preserves_class(self):
        reg = Registry("test")

        @reg.register
        class Original:
            """Docstring."""

            pass

        assert Original.__name__ == "Original"
        assert Original.__doc__ == "Docstring."


class TestGlobalRegistries:
    def test_all_registries_exist(self):
        from simpleml import DATASETS, LOSSES, METRICS, MODELS, OPTIMIZERS

        for reg in [MODELS, LOSSES, DATASETS, OPTIMIZERS, METRICS]:
            assert isinstance(reg, Registry)
