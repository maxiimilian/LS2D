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
The raw ERA5 dataset: the common output of all readers (CDS/MARS, Google ARCO).

    - Variables named by their registry key: `ml:t`, `pl:z`, `sfc:sp`, ...
    - dims `(time, level, latitude, longitude)`, `(time, pressure_level, latitude, longitude)`,
      or `(time, latitude, longitude)`.
    - `level` from surface to top (no coordinate), `pressure_level` [Pa] from surface to top,
      `latitude` south to north, `longitude` west to east in -180..180 (unless crossing the date line).
    - Attributes `central_lat`, `central_lon`, `source`.
"""

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
from ls2d.core.logger import logger

_level_dims = {'ml': 'level', 'pl': 'pressure_level'}


def missing_fields(ds, fields):
    """
    ERA5 fields (by short name) not present in `ds`.
    """
    return [f for f in fields if f.name not in ds.variables]


def fields_to_download(nc_file, fields, candidates):
    """
    ERA5 fields to (re-)download for `nc_file`: empty if the file exists and contains all `fields`.
    When fields are missing, the `candidates` already in the file are downloaded again as well,
    so that the new file replaces the old one without losing anything.
    """
    import os
    import netCDF4 as nc4

    if not os.path.isfile(nc_file):
        return list(fields)

    with nc4.Dataset(nc_file) as nc:
        present = set(nc.variables)

    if all(f.name in present for f in fields):
        return []

    return list(fields) + [f for f in candidates if f.name in present and f not in fields]


def group_by_levtype(fields):
    groups = {}
    for f in fields:
        groups.setdefault(f.levtype, []).append(f)
    return groups


def standardize(datasets, fields, attrs, pressure_level_dim='pressure_level'):
    """
    Combine per-levtype datasets, as read from ERA5 NetCDF files (short names, levels top->bottom,
    latitude north->south, pressure levels in hPa), into one raw dataset.

    Arguments:
        datasets : dict
            levtype -> xarray.Dataset. The same dataset can be used for several levtypes.
        fields : list of Era5Field
            Fields to put in the raw dataset.
        attrs : dict
            Global attributes, should include `central_lat` and `central_lon`.
        pressure_level_dim : str
            Name of the pressure level dimension in the `pl` dataset.
    """

    out = []
    for levtype, flds in group_by_levtype(fields).items():
        ds = datasets[levtype]
        ds = ds[[f.name for f in flds]]

        if levtype == 'ml':
            ds = ds.isel(level=slice(None, None, -1)).drop_vars('level', errors='ignore')
        elif levtype == 'pl':
            if pressure_level_dim != 'pressure_level':
                ds = ds.rename({pressure_level_dim: 'pressure_level'})
            ds = ds.sortby('pressure_level', ascending=False)
            ds = ds.assign_coords(pressure_level=('pressure_level', ds.pressure_level.values * 100.0, dict(units='Pa')))

        # Longitude -180..180, latitude south to north.
        lon = ds.longitude.values
        if np.any(lon > 180):
            lon_new = ((lon + 180) % 360) - 180
            if np.all(np.diff(lon_new) > 0):
                ds = ds.assign_coords(longitude=lon_new)
        ds = ds.sortby('latitude')

        ds = ds.rename({f.name: f.key for f in flds})
        for f in flds:
            ds[f.key].attrs.update(units=f.units or ds[f.key].attrs.get('units', ''), long_name=f.long_name)
        out.append(ds)

    # Horizontal grids should be identical; small floating point differences (e.g. after
    # re-gridding CDS surface/pressure level files) are removed.
    ref = out[0]
    for i, ds in enumerate(out[1:], start=1):
        for dim in ('latitude', 'longitude'):
            if ds.sizes[dim] != ref.sizes[dim] or not np.allclose(ds[dim], ref[dim], atol=1e-4):
                msg = f'ERA5 files have different {dim} grids:\n{ref[dim].values}\n{ds[dim].values}'
                logger.error(msg)
                raise ValueError(msg)
        out[i] = ds.assign_coords(latitude=ref.latitude, longitude=ref.longitude)

    try:
        raw = xr.merge(out, join='exact', combine_attrs='drop')
    except ValueError as e:
        msg = f'ERA5 files are not synchronised in time: {e}'
        logger.error(msg)
        raise ValueError(msg) from e

    raw = raw.drop_vars([c for c in raw.coords if c not in raw.dims])
    raw.attrs = dict(attrs)
    return raw
