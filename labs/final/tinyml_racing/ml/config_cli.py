"""Every leaf field of the config tree, reflected into a CLI flag.

`add_config_arguments` installs the flags a config declares, `configs_from_args`
reads them back. Everything a flag needs comes off the field itself: the
annotation gives the type and the arity, `Annotated[T, "..."]` gives the help
text, and the dataclass default is the flag default and is printed as one. So a
new config field is a flag with no edit here.
"""

import argparse
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Any, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    # Typeshed's "any dataclass" protocol, which is what `fields()` accepts. Only
    # a bound, and PEP 695 evaluates those lazily, so it costs nothing at runtime.
    from _typeshed import DataclassInstance

from tinyml_racing.ml.config import RacingEnvConfig, TrainConfig

# Flags mirror field names, and a nested config recurses with its field name as
# the prefix, so `PPOConfig.learning_rate` is `--ppo-learning-rate`.
_NO_FLAG = frozenset({"track_seed_range"})
# What argparse can parse from one token. Anything else (a path, an enum, a container of
# containers) has no obvious flag, and saying so while the parser is built surfaces it on
# any invocation, `--help` included.
_SCALARS = (int, float, str, bool)


def _flag(hint: Any) -> tuple[Any, str | None, int | str | None]:
    """`(type, help, nargs)` for one leaf annotation.

    `X | None` means the flag takes an `X` and None is what "unset" means.
    A `tuple[T, ...]` is variadic because its *length* is the setting, a
    network depth is chosen at the flag, while `tuple[T, T]` is a fixed-arity
    record, so a short `--track-length-range 800` is an argparse error rather
    than a wrong-length tuple that explodes deep inside track generation.
    """
    doc = None
    if get_origin(hint) is Annotated:
        hint, *extras = get_args(hint)
        doc = next((e for e in extras if isinstance(e, str)), None)
    if get_origin(hint) in (Union, UnionType):
        hint = next(a for a in get_args(hint) if a is not type(None))
    nargs: int | str | None = None
    if get_origin(hint) is tuple:
        args = get_args(hint)
        nargs = "+" if args[-1] is Ellipsis else len(args)
        hint = args[0]
    return hint, doc, nargs


def _shown(value: Any) -> str:
    """A default as a flag would be typed, not as Python repr()s it.

    Floats go through `%g`, so a value derived from the car reads 17.1335 rather
    than 17.133459161958886; large ints are grouped the way the source writes
    them, which `int()` also accepts back; None reads as what passing nothing
    means.
    """
    if value is None:
        return "unset"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:_}" if abs(value) >= 10_000 else str(value)
    if isinstance(value, tuple | list):
        return " ".join(_shown(v) for v in value)
    return str(value)


def _add_flags(group: argparse._ArgumentGroup, defaults: Any, prefix: str = "") -> None:
    """One flag per leaf field of `defaults`, recursing into nested configs.

    `defaults` is an instance, not a class: a `default_factory` field is worth
    printing as the value it resolved to.
    """
    hints = get_type_hints(type(defaults), include_extras=True)
    for f in fields(defaults):
        dest = prefix + f.name
        if dest in _NO_FLAG:
            continue
        default = getattr(defaults, f.name)
        if is_dataclass(default):
            # Two paths colliding on one dest is an argparse error at startup,
            # not a silent overwrite.
            _add_flags(group, default, prefix=f"{dest}_")
            continue
        hint = hints[f.name]
        leaf, doc, nargs = _flag(hint)
        if leaf not in _SCALARS:
            raise TypeError(
                f"config field {dest!r} is annotated {hint!r}, which is not a "
                f"flag this reflection builds; add it to _NO_FLAG or teach _flag its shape"
            )
        kwargs: dict[str, Any] = {
            "dest": dest,
            "default": list(default) if nargs else default,
            "help": " ".join(filter(None, (doc, f"(default: {_shown(default)})"))),
        }
        if leaf is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            # The type is the metavar: `--n-envs INT` says what one token has to
            # be, where argparse's default `N_ENVS` only repeats the flag, and a
            # fixed-arity field shows its arity as `FLOAT FLOAT`.
            kwargs.update(type=leaf, metavar=leaf.__name__.upper())
            if nargs:
                kwargs["nargs"] = nargs
        group.add_argument("--" + dest.replace("_", "-"), **kwargs)


def add_config_arguments[T: DataclassInstance](
    parser: argparse.ArgumentParser, cls: type[T], title: str
) -> None:
    _add_flags(parser.add_argument_group(title), cls())


def _from_args[T: DataclassInstance](cls: type[T], args: argparse.Namespace, prefix: str = "") -> T:
    """Instantiate `cls` from whatever prefixed flags the namespace carries.

    Every flag is named after the field it sets, so this is a lookup. Fields
    with no flag, and a group the caller never installed, keep their defaults.
    """
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for f in fields(defaults):
        dest = prefix + f.name
        default = getattr(defaults, f.name)
        if is_dataclass(default):
            kwargs[f.name] = _from_args(type(default), args, prefix=f"{dest}_")
        elif hasattr(args, dest):
            value = getattr(args, dest)
            # `nargs` hands back a list; the field is a tuple.
            kwargs[f.name] = tuple(value) if isinstance(default, tuple) else value
    return cls(**kwargs)


def configs_from_args(args: argparse.Namespace) -> tuple[RacingEnvConfig, TrainConfig]:
    """Both configs, filled from whatever flags the namespace carries."""
    return _from_args(RacingEnvConfig, args), _from_args(TrainConfig, args)
