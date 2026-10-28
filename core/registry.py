# registry.py
"""
Реестр алгоритмических блоков.

    @register("localizer", "srp_phat_siren")
    class SrpPhatSiren(BaseDOALocalizer): ...

Модули из пакета modules/ импортируются автоматически (discover_modules), поэтому новый файл
не нужно никуда прописывать — достаточно декоратора и имени в pipeline_config.json.
"""
import importlib
import pkgutil
from typing import Any, Callable, Dict, List, Type

from core.interfaces import (
    BaseBlock, BaseDistanceEstimator, BaseDOALocalizer, BaseEventDetector, BasePreProcessor,
)

KIND_BASES: Dict[str, Type[BaseBlock]] = {
    "pre_processor": BasePreProcessor,
    "detector": BaseEventDetector,
    "localizer": BaseDOALocalizer,
    "distance": BaseDistanceEstimator,
}

_REGISTRY: Dict[str, Dict[str, Type[BaseBlock]]] = {kind: {} for kind in KIND_BASES}
_discovered = False


def register(kind: str, name: str) -> Callable[[Type[BaseBlock]], Type[BaseBlock]]:
    if kind not in KIND_BASES:
        raise ValueError(f"Неизвестный тип блока '{kind}'. Допустимые: {list(KIND_BASES)}")

    def decorator(cls: Type[BaseBlock]) -> Type[BaseBlock]:
        if not issubclass(cls, KIND_BASES[kind]):
            raise TypeError(f"{cls.__name__} должен наследоваться от {KIND_BASES[kind].__name__}")
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Имя '{name}' ({kind}) уже занято классом {existing.__name__}")
        cls.name = name
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def discover_modules(package: str = "modules") -> None:
    """Импортирует все подмодули пакета, чтобы сработали декораторы @register."""
    global _discovered
    if _discovered:
        return
    pkg = importlib.import_module(package)
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        importlib.import_module(info.name)
    _discovered = True


def available(kind: str) -> List[str]:
    discover_modules()
    return sorted(_REGISTRY[kind])


def get_class(kind: str, name: str) -> Type[BaseBlock]:
    discover_modules()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"Блок '{name}' ({kind}) не зарегистрирован. Доступные: {available(kind)}") from None


def create(kind: str, name: str, **kwargs: Any) -> BaseBlock:
    return get_class(kind, name)(**kwargs)
