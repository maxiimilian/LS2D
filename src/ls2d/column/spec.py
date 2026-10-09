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
Validation of the raw dataset, which every source (ERA5, ...) must provide.
See `ls2d.forcing.raw` for the conventions.
"""

# Third party modules
import numpy as np

# LS2D modules
from ls2d.core.logger import logger
from ls2d.forcing.registry import Field
from ls2d.forcing.source import get_source

_dims = {
    'ml': ('time', 'level', 'latitude', 'longitude'),
    'pl': ('time', 'pressure_level', 'latitude', 'longitude'),
    'sfc': ('time', 'latitude', 'longitude'),
}


def _fail(msg):
    logger.error(msg)
    raise ValueError(msg)


def validate(ds):
    """
    Check if `ds` follows the raw dataset definition.
    """

    if 'ls2d_source' not in ds.attrs:
        _fail('Raw dataset: missing attribute "ls2d_source"')
    registry = get_source(ds.attrs['ls2d_source']).registry

    for name, da in ds.data_vars.items():
        if name not in registry or not isinstance(registry[name], Field):
            _fail(f'Raw dataset: "{name}" is not a registered field of source "{ds.attrs["ls2d_source"]}"')
        dims = _dims[registry[name].levtype]
        if da.dims != dims:
            _fail(f'Raw dataset: "{name}" has dims {da.dims}, expected {dims}')

    for attr in ('central_lat', 'central_lon'):
        if attr not in ds.attrs:
            _fail(f'Raw dataset: missing attribute "{attr}"')

    if np.any(np.diff(ds.latitude.values) <= 0):
        _fail('Raw dataset: latitude should be ascending (south to north)')
    if 'pressure_level' in ds.dims and np.any(np.diff(ds.pressure_level.values) >= 0):
        _fail('Raw dataset: pressure_level should be descending (surface to top)')
