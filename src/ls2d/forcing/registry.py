#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Original author: Bart van Stratum (WUR)
# Additional authors (refactoring): Maximilian Pierzyna (TU Delft), Claude (Anthropic)
#
# LS2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LS2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LS2D.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Registry of everything LS2D can download and compute.

There are three kinds of entries, which live in one namespace:

    - Raw fields of a data source (`Field`), with keys `{levtype}:{name}`, e.g. `ml:t`, `pl:z`, `sfc:sp`.
      These are the leaves: they are downloaded and read, never computed.
    - Field quantities (`Quantity`, stage='field'): computed pointwise on the 3D grid.
      Their `reduce` attribute ('mean', 'nearest', or a function) defines how they
      appear in the column dataset returned by `calculate_forcings()`.
    - Column quantities (`Quantity`, stage='column'): computed from the averaging
      sub-domain (including a halo for horizontal gradients), e.g. advective tendencies.

Each quantity declares the names it `requires`, and a pure function
`func(*required, ctx) -> xarray.DataArray`. `Registry.resolve()` walks these
dependencies, which determines both the evaluation order and which raw fields
need to be downloaded/read.

Registries can have a parent: names not found are looked up in the parent. LS2D uses
one `core` registry with the source-agnostic quantities, and one child registry per
data source (see `ls2d.forcing.source`), with the raw fields of that source and
the recipes that turn them into the standard quantities (see `ls2d.forcing.standard`).
A source can override core quantities by registering the same name.
"""

# Python modules
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

# LS2D modules
from ls2d.core.logger import logger
from ls2d.forcing.standard import STANDARD, check_dims

LEVTYPES = ('ml', 'pl', 'sfc')
STAGES = ('field', 'column')


class Field:
    """
    Base class of the raw fields of a data source. Subclasses are frozen dataclasses
    with at least the attributes `name` (name in the files), `levtype` ('ml', 'pl', or 'sfc'),
    `units`, and `long_name`, plus the source specific identifiers.
    """

    def __post_init__(self):
        if self.levtype not in LEVTYPES:
            raise ValueError(f'Field "{self.name}": levtype should be one of {LEVTYPES}, not "{self.levtype}"')

    @property
    def key(self):
        return f'{self.levtype}:{self.name}'


@dataclass(frozen=True)
class SimpleField(Field):
    """
    Raw field without source specific identifiers (or with a single one, `id`).
    """

    name: str
    levtype: str
    id: Optional[str] = None
    units: str = ''
    long_name: str = ''


Reducer = Union[str, Callable, None]


@dataclass(frozen=True)
class Quantity:
    """
    Derived quantity, computed by `func(*required, ctx)`.
    """

    name: str
    func: Callable
    requires: Tuple[str, ...] = ()
    stage: str = 'field'
    units: str = ''
    long_name: str = ''
    reduce: Reducer = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f'Quantity "{self.name}": stage should be one of {STAGES}, not "{self.stage}"')
        if self.stage == 'column' and self.reduce is not None:
            raise ValueError(f'Quantity "{self.name}": column quantities are already reduced, `reduce` not allowed')
        if ':' in self.name:
            raise ValueError(f'Quantity "{self.name}": ":" is reserved for raw field keys')


def _with_standard(q):
    """
    Fill in (and check) the metadata of a quantity with a standard name.
    """
    std = STANDARD.get(q.name)
    if std is None:
        return q
    if q.stage != 'field':
        raise ValueError(f'Quantity "{q.name}" has a standard name, and should be a field quantity')
    if q.units and q.units != std.units:
        raise ValueError(f'Quantity "{q.name}": units "{q.units}" differ from the standard units "{std.units}"')
    return dataclasses.replace(
        q,
        units=std.units,
        long_name=q.long_name or std.long_name,
        reduce=q.reduce if q.reduce is not None else std.reduce,
    )


class Registry:
    def __init__(self, name='', parent=None):
        self.name = name
        self.parent = parent
        self._fields = {}
        self._quantities = {}

    def __repr__(self):
        return f'Registry({self.name!r}, {len(self._fields)} fields, {len(self._quantities)} quantities)'

    #
    # Registration.
    #
    def add_field(self, f, replace=False):
        if not isinstance(f, Field):
            raise TypeError(f'{f!r} is not a `Field`')
        if f.key in self._fields and not replace:
            raise KeyError(f'Field "{f.key}" already registered in "{self.name}" (use `replace=True` to overwrite)')
        self._fields[f.key] = f
        return f

    def add(self, q, replace=False):
        if q.name in self._quantities and not replace:
            raise KeyError(f'Quantity "{q.name}" already registered in "{self.name}" (use `replace=True` to overwrite)')
        q = _with_standard(q)
        self._quantities[q.name] = q
        return q

    def quantity(self, name, requires=(), stage='field', units='', long_name='', reduce=None, replace=False, **attrs):
        """
        Decorator to register a quantity:

            @registry.quantity('thl', requires=('T', 'exn', 'ql'))
            def thl(T, exn, ql, ctx):
                return T / exn - Lv / (cpd * exn) * ql

        Quantities with a standard name (see `ls2d.forcing.standard`) inherit units,
        long name, and column reduction from the standard.
        """

        def decorator(func):
            self.add(Quantity(name, func, tuple(requires), stage, units, long_name, reduce, attrs), replace)
            return func

        return decorator

    #
    # Lookup (own entries first, then the parent).
    #
    def _lookup(self, name):
        if name in self._quantities:
            return self._quantities[name]
        if name in self._fields:
            return self._fields[name]
        if self.parent is not None:
            return self.parent._lookup(name)
        return None

    def __contains__(self, name):
        return self._lookup(name) is not None

    def __getitem__(self, name):
        entry = self._lookup(name)
        if entry is None:
            raise KeyError(self._unknown(name))
        return entry

    def _all_names(self):
        names = set(self._quantities) | set(self._fields)
        return names | (self.parent._all_names() if self.parent is not None else set())

    def _unknown(self, name):
        import difflib

        close = difflib.get_close_matches(name, sorted(self._all_names()), n=3)
        hint = f' Did you mean: {", ".join(close)}?' if close else ''
        where = f' in "{self.name}"' if self.name else ''
        return f'Unknown quantity or field "{name}"{where}.{hint}'

    def fields(self, levtype=None):
        """
        Raw fields (own and parent's), optionally only of `levtype`.
        """
        fields = dict(self.parent._all_fields()) if self.parent is not None else {}
        fields.update(self._fields)
        return [f for f in fields.values() if levtype is None or f.levtype == levtype]

    def _all_fields(self):
        return {f.key: f for f in self.fields()}

    def quantities(self, stage=None):
        """
        Quantities (own ones override the parent's), optionally only of `stage`.
        """
        qs = {q.name: q for q in self.parent.quantities()} if self.parent is not None else {}
        qs.update(self._quantities)
        return [q for q in qs.values() if stage is None or q.stage == stage]

    def column_names(self):
        """
        Names of all variables that `calculate_forcings()` can put in the column dataset.
        """
        return [q.name for q in self.quantities() if q.stage == 'column' or q.reduce is not None]

    #
    # Dependency resolution.
    #
    def resolve(self, names, available=()):
        """
        Return all entries needed to compute `names`, in evaluation order (dependencies first).
        Names in `available` are treated as given; their dependencies are not followed.
        """
        available = set(available)
        order = []
        state = {}  # name -> 'visiting' | 'done'

        def visit(name, path):
            if state.get(name) == 'done':
                return
            if state.get(name) == 'visiting':
                cycle = ' -> '.join(path[path.index(name) :] + [name])
                raise ValueError(f'Circular dependency: {cycle}')

            entry = self[name]  # raises for unknown names
            state[name] = 'visiting'
            if name not in available and isinstance(entry, Quantity):
                for req in entry.requires:
                    if req not in self:
                        raise KeyError(f'{self._unknown(req)} (required by "{name}")')
                    visit(req, path + [name])
            state[name] = 'done'
            order.append(entry)

        for name in names:
            visit(name, [])
        return order

    def required_fields(self, names, available=()):
        """
        Raw fields needed to compute `names`.
        """
        return [e for e in self.resolve(names, available) if isinstance(e, Field) and e.key not in available]

    def can_resolve(self, name, available):
        """
        True if `name` can be computed from `available` without any raw field that is not available.
        """
        try:
            return all(e.key in available for e in self.required_fields([name], available))
        except (KeyError, ValueError):
            return False

    def can_provide(self, name):
        """
        True if `name` can be computed from the registered raw fields.
        """
        return self.can_resolve(name, {f.key for f in self.fields()})

    def evaluate(self, names, inputs, ctx):
        """
        Compute `names` from `inputs` (dict name -> DataArray, for raw fields and/or
        already computed quantities). Returns dict with all computed and given values.
        Pure: `inputs` is not modified.
        """
        env = dict(inputs)
        for entry in self.resolve(names, available=env.keys()):
            key = entry.key if isinstance(entry, Field) else entry.name
            if key in env:
                continue
            if isinstance(entry, Field):
                msg = f'Field "{key}" is required, but not available in the input data.'
                logger.error(msg)
                raise KeyError(msg)
            try:
                value = entry.func(*[env[r] for r in entry.requires], ctx)
            except Exception as e:
                raise RuntimeError(f'Error while computing "{key}": {e}') from e
            check_dims(key, value)  # Standard quantities should have the standard dims.
            env[key] = value
        return env


# Source-agnostic quantities (`ls2d.forcing.derived`, `ls2d.forcing.column`).
core = Registry('core')
quantity = core.quantity
