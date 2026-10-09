#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Author: Bart van Stratum (WUR)
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

    - ERA5 fields (`Era5Field`), with keys `{levtype}:{name}`, e.g. `ml:t`, `pl:z`, `sfc:sp`.
      These are the leaves: they are downloaded and read, never computed.
    - Field quantities (`Quantity`, stage='field'): computed pointwise on the 3D ERA5 grid.
      Their `reduce` attribute ('mean', 'nearest', or a function) defines how they
      appear in the column dataset returned by `calculate_forcings()`.
    - Column quantities (`Quantity`, stage='column'): computed from the averaging
      sub-domain (including a halo for horizontal gradients), e.g. advective tendencies.

Each quantity declares the names it `requires`, and a pure function
`func(*required, ctx) -> xarray.DataArray`. `Registry.resolve()` walks these
dependencies, which determines both the evaluation order and which ERA5 fields
need to be downloaded/read.
"""

# Python modules
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

# LS2D modules
from ls2d.core.logger import logger

LEVTYPES = ('ml', 'pl', 'sfc')
STAGES = ('field', 'column')


@dataclass(frozen=True)
class Era5Field:
    """
    One raw ERA5 variable, with its names in the different data sources.

    Arguments:
        name : short name, as used in the NetCDF files (e.g. 't', 'sp').
        levtype : 'ml' (model levels), 'pl' (pressure levels), or 'sfc' (surface / single level).
        param : GRIB parameter ID, as used in MARS and CDS "complete" (model level) requests.
        cds : name in the CDS single/pressure level datasets (not needed for model level fields).
        arco : name in the Google ARCO-ERA5 Zarr stores (None if not available there).
    """

    name: str
    levtype: str
    param: str
    cds: Optional[str] = None
    arco: Optional[str] = None
    units: str = ''
    long_name: str = ''

    def __post_init__(self):
        if self.levtype not in LEVTYPES:
            raise ValueError(f'ERA5 field "{self.name}": levtype should be one of {LEVTYPES}, not "{self.levtype}"')

    @property
    def key(self):
        return f'{self.levtype}:{self.name}'


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
            raise ValueError(f'Quantity "{self.name}": ":" is reserved for ERA5 field keys')


class Registry:
    def __init__(self):
        self._era5 = {}
        self._quantities = {}

    #
    # Registration.
    #
    def add_era5_field(self, f, replace=False):
        if f.key in self._era5 and not replace:
            raise KeyError(f'ERA5 field "{f.key}" already registered (use `replace=True` to overwrite)')
        self._era5[f.key] = f
        return f

    def era5_field(self, name, levtype, param, cds=None, arco=None, units='', long_name='', replace=False):
        """
        Register a raw ERA5 field. See `Era5Field`.
        """
        return self.add_era5_field(Era5Field(name, levtype, param, cds, arco, units, long_name), replace)

    def add(self, q, replace=False):
        if q.name in self._quantities and not replace:
            raise KeyError(f'Quantity "{q.name}" already registered (use `replace=True` to overwrite)')
        self._quantities[q.name] = q
        return q

    def quantity(self, name, requires=(), stage='field', units='', long_name='', reduce=None, replace=False, **attrs):
        """
        Decorator to register a quantity:

            @registry.quantity('thl', requires=('T', 'exn', 'ql'), units='K', reduce='mean')
            def thl(T, exn, ql, ctx):
                return T / exn - ctx.ifs.Lv / (ctx.ifs.cpd * exn) * ql
        """

        def decorator(func):
            self.add(Quantity(name, func, tuple(requires), stage, units, long_name, reduce, attrs), replace)
            return func

        return decorator

    #
    # Lookup.
    #
    def __contains__(self, name):
        return name in self._quantities or name in self._era5

    def __getitem__(self, name):
        if name in self._quantities:
            return self._quantities[name]
        if name in self._era5:
            return self._era5[name]
        raise KeyError(self._unknown(name))

    def _unknown(self, name):
        import difflib

        close = difflib.get_close_matches(name, list(self._quantities) + list(self._era5), n=3)
        hint = f' Did you mean: {", ".join(close)}?' if close else ''
        return f'Unknown quantity or ERA5 field "{name}".{hint}'

    def era5_fields(self, levtype=None):
        return [f for f in self._era5.values() if levtype is None or f.levtype == levtype]

    def quantities(self, stage=None):
        return [q for q in self._quantities.values() if stage is None or q.stage == stage]

    def column_names(self):
        """
        Names of all variables that `calculate_forcings()` can put in the column dataset.
        """
        return [q.name for q in self._quantities.values() if q.stage == 'column' or q.reduce is not None]

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

    def required_era5_fields(self, names, available=()):
        """
        ERA5 fields needed to compute `names`.
        """
        return [e for e in self.resolve(names, available) if isinstance(e, Era5Field) and e.key not in available]

    def can_resolve(self, name, available):
        """
        True if `name` can be computed from `available` without any ERA5 field that is not available.
        """
        try:
            return all(e.key in available for e in self.required_era5_fields([name], available))
        except (KeyError, ValueError):
            return False

    def evaluate(self, names, inputs, ctx):
        """
        Compute `names` from `inputs` (dict name -> DataArray, for ERA5 fields and/or
        already computed quantities). Returns dict with all computed and given values.
        Pure: `inputs` is not modified.
        """
        env = dict(inputs)
        for entry in self.resolve(names, available=env.keys()):
            key = entry.key if isinstance(entry, Era5Field) else entry.name
            if key in env:
                continue
            if isinstance(entry, Era5Field):
                msg = f'ERA5 field "{key}" is required, but not available in the input data.'
                logger.error(msg)
                raise KeyError(msg)
            try:
                env[key] = entry.func(*[env[r] for r in entry.requires], ctx)
            except Exception as e:
                raise RuntimeError(f'Error while computing "{key}": {e}') from e
        return env


# Default, global registry. Populated by `ls2d.forcing.era5_fields`, `.fields`, `.column`.
registry = Registry()
era5_field = registry.era5_field
quantity = registry.quantity
