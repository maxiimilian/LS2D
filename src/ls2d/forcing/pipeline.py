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
Stateless pipeline:

    fields = required_era5_fields(outputs)           # which ERA5 fields to download/read
    raw    = read_era5(settings, outputs)             # raw ERA5 dataset (see `ls2d.forcing.raw`)
    column = calculate_forcings(raw, n_av, method)    # column dataset (time, level)
    les    = get_les_input(column, z)                 # LES/SCM input on height grid `z`

`outputs` can contain LES output names (see `les_outputs()`) and/or column variable
names (see `registry.column_names()`). Default: all LES outputs.
"""

# Third party modules
import xarray as xr

# LS2D modules
import ls2d.core.spatial_tools as spatial
from ls2d.core.logger import logger
from ls2d.forcing.context import Context, Domain, format_latlon
from ls2d.forcing.les import les_outputs, les_sources
from ls2d.forcing.registry import registry as default_registry


def column_names(outputs=None, registry=default_registry):
    """
    Column variables needed for `outputs` (LES output and/or column variable names).
    """
    if outputs is None:
        return les_sources()

    available = set(registry.column_names())
    names = []
    for o in outputs:
        if o in les_outputs():
            names.extend(les_sources([o]))
        elif o in available:
            names.append(o)
        elif o in registry:
            raise ValueError(
                f'"{o}" is a field quantity or ERA5 field without `reduce`; it can not be used as column output.'
            )
        else:
            raise KeyError(registry._unknown(o))
    return list(dict.fromkeys(names))


def required_era5_fields(outputs=None, registry=default_registry):
    """
    ERA5 fields needed to compute `outputs` (default: all LES outputs).
    """
    return registry.required_era5_fields(column_names(outputs, registry))


def _inputs(ds):
    return {name: ds[name] for name in ds.data_vars}


def compute_fields(raw, names=None, registry=default_registry):
    """
    Compute field quantities on the full 3D grid.

    Arguments:
        raw : xarray.Dataset
            Raw ERA5 dataset, from e.g. `ls2d.read_era5()`.
        names : list of str, optional
            Field quantities to compute. Default: all field quantities that
            can be computed from the fields in `raw`.

    Returns:
        xarray.Dataset with the field quantities.
    """
    inputs = _inputs(raw)
    if names is None:
        names = [q.name for q in registry.quantities('field') if registry.can_resolve(q.name, inputs)]

    ctx = Context(raw.attrs['central_lat'], raw.attrs['central_lon'])
    env = registry.evaluate(names, inputs, ctx)

    ds = xr.Dataset({name: env[name] for name in names}, attrs=dict(raw.attrs))
    for name in names:
        q = registry[name]
        ds[name].attrs.update(units=q.units, long_name=q.long_name)
    return ds


def calculate_forcings(raw, n_av=0, method='2nd', outputs=None, registry=default_registry):
    """
    Calculate the large-scale forcings and the (area averaged) column profiles.

    Arguments:
        raw : xarray.Dataset
            Raw ERA5 dataset (from `ls2d.read_era5()`), or a dataset with already computed
            field quantities (from `ls2d.compute_fields()`). Attributes `central_lat` and
            `central_lon` define the location of the column.
        n_av : int
            Number of grid points (+/-) over which the fields and forcings are averaged.
        method : str
            '2nd' or '4th' order horizontal gradients.
        outputs : list of str, optional
            LES outputs and/or column variables to compute. Default: all LES outputs.

    Returns:
        xarray.Dataset with the column variables, dims (time, [level | level_half | soil_layer]).
    """

    logger.info('Calculating large-scale forcings')

    names = column_names(outputs, registry)
    clat, clon = raw.attrs['central_lat'], raw.attrs['central_lon']

    domain = Domain.from_grid(raw.latitude.values, raw.longitude.values, clat, clon, n_av, method)
    distance = spatial.haversine(domain.lon, domain.lat, clon, clat)
    logger.info(
        f'Averaging {domain.area} @ {format_latlon(domain.lat, domain.lon)} '
        f'(requested: {format_latlon(clat, clon)}, distance = {distance / 1000:.1f} km)'
    )

    ctx = Context(clat, clon, n_av, method, domain)
    env = registry.evaluate(names, _inputs(domain.subset(raw)), ctx)

    column = {}
    for name in names:
        q = registry[name]
        if q.stage == 'column':
            da = env[name]
        elif q.reduce == 'mean':
            da = ctx.mean(env[name])
        elif q.reduce == 'nearest':
            da = ctx.nearest(env[name])
        else:
            da = q.reduce(env[name], ctx)
        da = da.drop_vars([c for c in da.coords if c not in da.dims and c != 'z_soil'])
        column[name] = da.assign_attrs(units=q.units, long_name=q.long_name)

    return xr.Dataset(
        column,
        attrs={
            'central_lat': clat,
            'central_lon': clon,
            'lat': domain.lat,
            'lon': domain.lon,
            'n_av': n_av,
            'method': method,
            'fc': ctx.fc,
            'area': domain.area,
            'source': raw.attrs.get('source', 'unknown'),
        },
    )
