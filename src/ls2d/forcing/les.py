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
LES/SCM input: which column variables end up in the output of `get_les_input()`, and how.
"""

# Python modules
from dataclasses import dataclass
from typing import Optional

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
from ls2d.core.logger import logger


@dataclass(frozen=True)
class LesVar:
    """
    Arguments:
        name : name in the LES input dataset.
        src : name of the column variable (see `ls2d.forcing.registry.Registry.column_names()`).
        interp : interpolate from model levels to the LES height grid `z`. Otherwise, the
            variable is copied, with dims `level` -> `lay`, `level_half` -> `lev`, `soil_layer` -> `zs`.
        scale : multiply by this factor (unit conversion).
        scheme : land surface scheme the variable belongs to (e.g. 'HTESSEL'), None if generic.
    """

    name: str
    src: str
    long_name: str
    units: str
    interp: bool = False
    scale: float = 1.0
    scheme: Optional[str] = None


_les_outputs = {}


def les_output(name, src, long_name, units, interp=False, scale=1.0, scheme=None, replace=False):
    """
    Register an LES/SCM output variable.
    """
    if name in _les_outputs and not replace:
        raise KeyError(f'LES output "{name}" already registered (use `replace=True` to overwrite)')
    _les_outputs[name] = LesVar(name, src, long_name, units, interp, scale, scheme)
    return _les_outputs[name]


def les_outputs():
    """
    Names of all registered LES/SCM output variables (the default set).
    """
    return list(_les_outputs)


def les_sources(outputs=None):
    """
    Column variables required for LES/SCM `outputs` (default: all). Includes the full
    level height `z` if any of the outputs is interpolated to the LES grid.
    """
    outputs = les_outputs() if outputs is None else outputs
    unknown = [o for o in outputs if o not in _les_outputs]
    if unknown:
        raise KeyError(f'Unknown LES output(s): {unknown}. Available: {les_outputs()}')
    sources = [_les_outputs[o].src for o in outputs]
    if any(_les_outputs[o].interp for o in outputs):
        sources.append('z')
    return list(dict.fromkeys(sources))


# fmt: off
les_output('thl',         'thl',         'liquid water potential temperature', 'K', interp=True)
les_output('qt',          'qt',          'total specific humidity', 'kg kg-1', interp=True)
les_output('u',           'u',           'zonal wind component', 'm s-1', interp=True)
les_output('v',           'v',           'meridional wind component', 'm s-1', interp=True)
les_output('wls',         'w',           'vertical wind component', 'm s-1', interp=True)
les_output('p',           'p',           'air pressure', 'Pa', interp=True)
les_output('dtthl_advec', 'dtthl_advec', 'advective tendency liquid water potential temperature', 'K s-1', interp=True)
les_output('dtqt_advec',  'dtqt_advec',  'advective tendency total specific humidity', 'kg kg-1 s-1', interp=True)
les_output('dtu_advec',   'dtu_advec',   'advective tendency zonal wind', 'm s-2', interp=True)
les_output('dtv_advec',   'dtv_advec',   'advective tendency meridional wind', 'm s-2', interp=True)
les_output('ug',          'ug',          'geostrophic wind component zonal wind', 'm s-1', interp=True)
les_output('vg',          'vg',          'geostrophic wind component meridional wind', 'm s-1', interp=True)
les_output('o3',          'o3',          'ozone volume mixing ratio', 'ppmv', interp=True, scale=1e6)

# Radiation background profiles.
les_output('z_lay',       'z',           'Full level heights radiation', 'm')
les_output('z_lev',       'zh',          'Half level heights radiation', 'm')
les_output('p_lay',       'p',           'full level pressure radiation', 'Pa')
les_output('p_lev',       'ph',          'half level pressure radiation', 'Pa')
les_output('t_lay',       'T',           'full level temperature radiation', 'K')
les_output('t_lev',       'Th',          'half level temperature radiation', 'K')
les_output('h2o_lay',     'h2o',         'moisture volume mixing ratio', '')
les_output('o3_lay',      'o3',          'ozone volume mixing ratio radiation', 'ppmv', scale=1e6)

# Surface.
les_output('ps',          'ps',          'surface pressure', 'Pa')
les_output('ts',          'ts',          'surface (skin) temperature', 'K')
les_output('sst',         'sst',         'sea surface temperature', 'K')
les_output('wth',         'wth',         'surface sensible heat flux', 'K m s-1')
les_output('wq',          'wq',          'surface latent heat flux', 'kg kg-1 m s-1')
les_output('z0m',         'z0m',         'roughness length momentum', 'm')
les_output('z0h',         'z0h',         'roughness length scalars', 'm')

# Land surface.
les_output('lai_low_veg',        'lai_low',            'LAI low vegetation', '-')
les_output('lai_high_veg',       'lai_high',           'LAI high vegetation', '-')
les_output('c_low_veg',          'c_low_veg',          'fraction low vegetation', '-')
les_output('c_high_veg',         'c_high_veg',         'fraction high vegetation', '-')
les_output('t_soil',             't_soil',             'soil temperature', 'K')
les_output('theta_soil',         'theta_soil',         'soil moisture content', 'm3 m-3')
les_output('type_soil',          'type_soil',          'ECMWF soil type (Fortran indexing!)', '-', scheme='HTESSEL')
les_output('type_low_veg',       'type_low_veg',       'ECMWF low vegetation type (Fortran indexing!)', '-', scheme='HTESSEL')
les_output('type_high_veg',      'type_high_veg',      'ECMWF high vegetation type (Fortran indexing!)', '-', scheme='HTESSEL')
les_output('root_frac_low_veg',  'root_frac_low_veg',  'root fraction low vegetation', '-', scheme='HTESSEL')
les_output('root_frac_high_veg', 'root_frac_high_veg', 'root fraction high vegetation', '-', scheme='HTESSEL')
# fmt: on

_rename = {'level': 'lay', 'level_half': 'lev', 'soil_layer': 'zs'}

reference = (
    'van Stratum et al. (2023). The benefits and challenges of downscaling a global reanalysis with '
    'doubly-periodic large-eddy simulations. JAMES, https://doi.org/10.1029/2023MS003750'
)


def get_les_input(column, z, outputs=None):
    """
    Interpolate the column dataset from `ls2d.calculate_forcings()` to the
    LES/SCM height grid, and return xarray.Dataset with the LES/SCM input.

    Arguments:
        column : xarray.Dataset
            Column dataset from `ls2d.calculate_forcings()`.
        z : np.ndarray
            Full level heights LES grid (m).
        outputs : list of str, optional
            LES variables to include (see `ls2d.les_outputs()`). Default: all variables that
            are available in `column`; when given explicitly, missing variables raise an error.

    The result only depends on the content of `column`, not on the data source.
    """

    if outputs is None:
        outputs = [o for o in les_outputs() if _les_outputs[o].src in column]
        skipped = [o for o in les_outputs() if o not in outputs]
        if skipped:
            logger.debug(f'Not in column dataset, skipped LES outputs: {skipped}')
    else:
        missing = [s for s in les_sources(outputs) if s not in column]
        if missing:
            msg = (
                f'Column dataset misses variable(s) {missing}. '
                f'Calculate them with `ls2d.calculate_forcings(..., outputs=...)` first.'
            )
            logger.error(msg)
            raise KeyError(msg)

    z = np.asarray(z, dtype=float)
    z_da = xr.DataArray(z, dims='z', coords=dict(z=z), attrs=dict(long_name='full level height LES', units='m'))

    needs_interp = any(_les_outputs[o].interp for o in outputs)
    if needs_interp and 'z' not in column:
        raise KeyError('Column dataset misses full level height `z`, needed for the interpolation to the LES grid.')

    def to_les(da):
        return xr.apply_ufunc(
            np.interp, z_da, column['z'], da,
            input_core_dims=[['z'], ['level'], ['level']], output_core_dims=[['z']], vectorize=True,
        )  # fmt: skip

    out = xr.Dataset()
    for name in outputs:
        var = _les_outputs[name]
        da = column[var.src]
        da = to_les(da) if var.interp else da.rename({k: v for k, v in _rename.items() if k in da.dims})
        if var.scale != 1.0:
            da = da * var.scale
        out[name] = da
        out[name].attrs = dict(long_name=var.long_name, units=var.units)
        if var.scheme is not None:
            out[name].attrs['land_surface_scheme'] = var.scheme

    # Coordinates: drop all non-dimension coordinates, and set the LS2D ones.
    out = out.drop_vars([c for c in out.coords if c not in out.dims])
    if 'zs' in out.dims:
        out = out.assign_coords(zs=('zs', column.z_soil.values, dict(long_name='full level depth soil', units='m')))
    for dim in ('lay', 'lev'):
        if dim in out.dims:
            out = out.assign_coords({dim: np.arange(out.sizes[dim])})
    if 'z' in out.dims:
        out['z'].attrs = z_da.attrs

    out['time_sec'] = ('time', (column.time.values - column.time.values[0]) / np.timedelta64(1, 's'))
    out['time_sec'].attrs = dict(long_name='seconds since start of experiment', units='s')

    out.attrs = {
        'fc': column.attrs.get('fc'),
        'central_lon': column.attrs.get('central_lon'),
        'central_lat': column.attrs.get('central_lat'),
        'area': f'{column.attrs.get("area")} spatial average',
        'source': f'{column.attrs.get("source", "unknown")} + (LS)²D',
        'ls2d_source': column.attrs.get('ls2d_source', 'unknown'),
        'description': 'Generated by (LS)²D: https://github.com/LS2D & https://pypi.org/project/ls2d)',
        'reference': reference,
    }

    return out
