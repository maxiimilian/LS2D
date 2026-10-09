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
The standard quantities: the contract between the data sources (ERA5, GFS, ...) and the
source-agnostic part of LS2D (`ls2d.forcing.derived`, `ls2d.forcing.column`, `ls2d.forcing.les`).

A source provides the `source` quantities from its raw fields; LS2D derives everything
else from those. A quantity registered under a standard name inherits the metadata below
(units, long name, column reduction), and is checked against it: registering a standard
quantity with different units is an error, and computed values must have the standard dims.

Conventions:
    - SI units, fluxes positive upward.
    - `level` from surface to top, with `p` decreasing. Any terrain-following vertical
      coordinate is fine (hybrid model levels, sigma levels, ...); `level_half` has one more level,
      the first one at the surface.
    - `pressure_level` [Pa] from surface to top.
    - `soil_layer` from top to bottom, with coordinate `z_soil` [m, negative].
    - regular latitude/longitude grid, latitude south to north.
"""

# Python modules
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

# Third party modules
import xarray as xr

# LS2D modules
from ls2d.core.logger import logger

d3 = ('time', 'level', 'latitude', 'longitude')
d3h = ('time', 'level_half', 'latitude', 'longitude')
d3p = ('time', 'pressure_level', 'latitude', 'longitude')
d3s = ('time', 'soil_layer', 'latitude', 'longitude')
d3s_static = ('soil_layer', 'latitude', 'longitude')
d2 = ('time', 'latitude', 'longitude')


def nearest_type(da, ctx, log=False):
    """
    Land surface type at the central column (first time step). Sea points are 1e9.
    """
    value = int(ctx.nearest(da).isel(time=0))
    if log:
        if value == int(1e9):
            logger.warning('Selected grid point is water/sea! Setting vegetation/soil indexes to 1e9.')
        else:
            logger.debug('Selected grid point is over land.')
    return xr.DataArray(value)


def nearest_type_log(da, ctx):
    return nearest_type(da, ctx, log=True)


@dataclass(frozen=True)
class StandardVar:
    """
    Arguments:
        dims, units, long_name : definition of the quantity.
        reduce : how the quantity appears in the column dataset ('mean', 'nearest', function, or None = not at all).
        provider : 'source' (each source provides it) or 'core' (derived by LS2D, sources may override).
        required : sources must be able to provide it (only for provider='source').
        scheme : land surface scheme the quantity belongs to (e.g. 'HTESSEL'), None if generic.
    """

    dims: Tuple[str, ...]
    units: str
    long_name: str
    reduce: Union[str, Callable, None] = 'mean'
    provider: str = 'source'
    required: bool = False
    scheme: Optional[str] = None


S = StandardVar

# fmt: off
STANDARD = {
    # Provided by every source.
    'T':      S(d3,  'K',        'absolute temperature', required=True),
    'qv':     S(d3,  'kg kg-1',  'specific humidity', reduce=None, required=True),
    'ql':     S(d3,  'kg kg-1',  'total condensate (liquid + ice) specific humidity', reduce=None, required=True),
    'u':      S(d3,  'm s-1',    'zonal wind', required=True),
    'v':      S(d3,  'm s-1',    'meridional wind', required=True),
    'omega':  S(d3,  'Pa s-1',   'pressure vertical velocity', reduce=None, required=True),
    'p':      S(d3,  'Pa',       'full level pressure', required=True),  # Source provides `p` or `ph`.
    'ph':     S(d3h, 'Pa',       'half level pressure', required=True),
    'phi_p':  S(d3p, 'm2 s-2',   'geopotential on pressure levels', reduce=None, required=True),
    'ps':     S(d2,  'Pa',       'surface pressure', required=True),
    'ts':     S(d2,  'K',        'surface (skin) temperature', required=True),
    'wth':    S(d2,  'K m s-1',  'surface kinematic sensible heat flux (positive upward)', required=True),
    'wq':     S(d2,  'kg kg-1 m s-1', 'surface kinematic moisture flux (positive upward)', required=True),

    # Optional, generic.
    'o3':         S(d3,  'mol mol-1', 'ozone volume mixing ratio'),
    'sst':        S(d2,  'K',         'sea surface temperature'),
    'z0m':        S(d2,  'm',         'roughness length momentum'),
    'z0h':        S(d2,  'm',         'roughness length scalars'),
    'lai_low':    S(d2,  '-',         'leaf area index low vegetation'),
    'lai_high':   S(d2,  '-',         'leaf area index high vegetation'),
    'c_low_veg':  S(d2,  '-',         'fraction low vegetation'),
    'c_high_veg': S(d2,  '-',         'fraction high vegetation'),
    't_soil':     S(d3s, 'K',         'soil temperature'),
    'theta_soil': S(d3s, 'm3 m-3',    'soil moisture content'),

    # Optional, specific for a land surface scheme.
    'type_soil':          S(d2, '-', 'ECMWF soil type (Fortran indexing!), 1e9 over sea', nearest_type_log, scheme='HTESSEL'),
    'type_low_veg':       S(d2, '-', 'ECMWF low vegetation type (Fortran indexing!), 1e9 over sea', nearest_type, scheme='HTESSEL'),
    'type_high_veg':      S(d2, '-', 'ECMWF high vegetation type (Fortran indexing!), 1e9 over sea', nearest_type, scheme='HTESSEL'),
    'root_frac_low_veg':  S(d3s_static, '-', 'root fraction low vegetation (-1 over sea)', 'nearest', scheme='HTESSEL'),
    'root_frac_high_veg': S(d3s_static, '-', 'root fraction high vegetation (-1 over sea)', 'nearest', scheme='HTESSEL'),

    # Derived by LS2D (`ls2d.forcing.derived`) from the quantities above.
    'qt':   S(d3,  'kg kg-1',   'total specific humidity', provider='core'),
    'Tv':   S(d3,  'K',         'virtual temperature', reduce=None, provider='core'),
    'zh':   S(d3h, 'm',         'half level height above surface', provider='core'),
    'z':    S(d3,  'm',         'full level height above surface', provider='core'),
    'exn':  S(d3,  '-',         'Exner function', reduce=None, provider='core'),
    'thl':  S(d3,  'K',         'liquid water potential temperature', provider='core'),
    'rho':  S(d3,  'kg m-3',    'density', reduce=None, provider='core'),
    'w':    S(d3,  'm s-1',     'vertical wind', provider='core'),
    'h2o':  S(d3,  'mol mol-1', 'moisture volume mixing ratio', provider='core'),
    'rhos': S(d2,  'kg m-3',    'surface density (using lowest level humidity)', reduce=None, provider='core'),
}
# fmt: on


def required_names():
    """
    Standard quantities that every source must be able to provide (directly, or via the core fallbacks).
    """
    return [n for n, s in STANDARD.items() if s.required or s.provider == 'core']


def check_dims(name, da):
    """
    Check that a computed standard quantity has the standard dims.
    """
    std = STANDARD.get(name)
    if std is not None and tuple(da.dims) != std.dims:
        msg = f'Standard quantity "{name}" has dims {tuple(da.dims)}, expected {std.dims}'
        logger.error(msg)
        raise ValueError(msg)
