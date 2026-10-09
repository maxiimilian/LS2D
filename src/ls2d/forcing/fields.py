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
Field quantities: computed pointwise on the 3D ERA5 grid, from raw ERA5 fields.

Conventions (input and output):
    - SI units.
    - dims `(time, level, latitude, longitude)` for full levels, `level_half` for half levels,
      `pressure_level` for pressure levels, `soil_layer` for the soil.
    - levels and pressure levels from surface to top, latitude south to north.

`reduce` defines how a quantity appears in the column dataset (see `ls2d.forcing.column`).
"""

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
import ls2d.ecmwf.htessel as htessel
from ls2d.core.logger import logger
from ls2d.forcing.registry import quantity

# Molar mass ratio dry air / ozone.
md_mo3 = 28.9644 / 47.9982

d3 = ('time', 'level', 'latitude', 'longitude')
d3h = ('time', 'level_half', 'latitude', 'longitude')


def _like(values, dims, *templates):
    """
    DataArray from numpy `values`, with the coordinates of `templates` for `dims`.
    """
    coords = {}
    for t in templates:
        for name, c in t.coords.items():
            if set(c.dims) <= set(dims) and name not in coords:
                coords[name] = c
    return xr.DataArray(values, dims=dims, coords=coords)


def _nearest_type(da, ctx, log=False):
    """
    Vegetation/soil type at the central column (first time step). Sea points are set to 1e9 (see `type_soil`).
    """
    value = int(ctx.nearest(da).isel(time=0))
    if log:
        if value == int(1e9):
            logger.warning('Selected grid point is water/sea! Setting vegetation/soil indexes to 1e9.')
        else:
            logger.debug('Selected grid point is over land.')
    return xr.DataArray(value)


#
# Thermodynamics on model levels.
#
@quantity('T', requires=('ml:t',), units='K', long_name='absolute temperature', reduce='mean')
def T(t, ctx):
    return t


@quantity('ql', requires=('ml:clwc', 'ml:ciwc', 'ml:crwc', 'ml:cswc'), units='kg kg-1',
          long_name='total liquid + ice specific humidity')  # fmt: skip
def ql(clwc, ciwc, crwc, cswc, ctx):
    return clwc + ciwc + crwc + cswc


@quantity('qt', requires=('ml:q', 'ql'), units='kg kg-1', long_name='total specific humidity', reduce='mean')
def qt(q, ql, ctx):
    return q + ql


@quantity('Tv', requires=('ml:t', 'ml:q', 'ml:clwc', 'ml:ciwc', 'ml:crwc', 'ml:cswc'), units='K',
          long_name='virtual temperature')  # fmt: skip
def Tv(t, q, clwc, ciwc, crwc, cswc, ctx):
    return ctx.ifs.calc_virtual_temp(t, q, clwc, ciwc, crwc, cswc)


@quantity('ph', requires=('sfc:sp',), units='Pa', long_name='half level pressure', reduce='mean')
def ph(sp, ctx):
    ifs = ctx.ifs
    ps = sp.values
    p = ifs.a[None, :, None, None] + ifs.b[None, :, None, None] * ps[:, None]
    p[:, -1] = 0.34  # As in `IFS_tools.calc_half_level_pressure()`
    return _like(p, d3h, sp)


@quantity('zh', requires=('ph', 'Tv'), units='m', long_name='half level height above surface', reduce='mean')
def zh(ph, Tv, ctx):
    p = ph.values
    dz = -ctx.ifs.Rd * Tv.values * np.log(p[:, 1:] / p[:, :-1]) / ctx.ifs.grav
    z = np.concatenate((np.zeros_like(p[:, :1]), np.cumsum(dz, axis=1)), axis=1)
    return _like(z, d3h, ph)


@quantity('p', requires=('ph',), units='Pa', long_name='full level pressure', reduce='mean')
def p(ph, ctx):
    p = ph.values
    return _like(0.5 * (p[:, 1:] + p[:, :-1]), d3, ph)


@quantity('z', requires=('zh',), units='m', long_name='full level height above surface', reduce='mean')
def z(zh, ctx):
    z = zh.values
    return _like(0.5 * (z[:, 1:] + z[:, :-1]), d3, zh)


@quantity('exn', requires=('p',), units='-', long_name='Exner function')
def exn(p, ctx):
    return ctx.ifs.calc_exner(p)


@quantity('thl', requires=('ml:t', 'exn', 'ql'), units='K', long_name='liquid water potential temperature',
          reduce='mean')  # fmt: skip
def thl(t, exn, ql, ctx):
    return t / exn - ctx.ifs.Lv / (ctx.ifs.cpd * exn) * ql


@quantity('rho', requires=('p', 'Tv'), units='kg m-3', long_name='density')
def rho(p, Tv, ctx):
    return p / (ctx.ifs.Rd * Tv)


@quantity('u', requires=('ml:u',), units='m s-1', long_name='zonal wind', reduce='mean')
def u(u, ctx):
    return u


@quantity('v', requires=('ml:v',), units='m s-1', long_name='meridional wind', reduce='mean')
def v(v, ctx):
    return v


@quantity('w', requires=('ml:w', 'rho'), units='m s-1', long_name='vertical wind', reduce='mean')
def w(w, rho, ctx):
    return -w / (rho * ctx.ifs.grav)


@quantity('h2o', requires=('ml:q', 'qt'), units='mol mol-1', long_name='moisture volume mixing ratio', reduce='mean')
def h2o(q, qt, ctx):
    return q / ((ctx.ifs.Rd / ctx.ifs.Rv) * (1 - qt))


@quantity('o3', requires=('ml:o3',), units='mol mol-1', long_name='ozone volume mixing ratio', reduce='mean')
def o3(o3, ctx):
    return o3 * md_mo3


@quantity('phi_p', requires=('pl:z',), units='m2 s-2', long_name='geopotential on pressure levels')
def phi_p(z, ctx):
    return z


#
# Surface.
#
@quantity('ps', requires=('sfc:sp',), units='Pa', long_name='surface pressure', reduce='mean')
def ps(sp, ctx):
    return sp


@quantity('ts', requires=('sfc:skt',), units='K', long_name='surface (skin) temperature', reduce='mean')
def ts(skt, ctx):
    return skt


@quantity('sst', requires=('sfc:sst',), units='K', long_name='sea surface temperature', reduce='mean')
def sst(sst, ctx):
    return sst


@quantity('rhos', requires=('sfc:sp', 'sfc:skt', 'ml:q'), units='kg m-3',
          long_name='surface density (using lowest model level humidity)')  # fmt: skip
def rhos(sp, skt, q, ctx):
    return sp / (ctx.ifs.Rd * ctx.ifs.calc_virtual_temp(skt, q.isel(level=0, drop=True)))


@quantity('wth', requires=('sfc:ishf', 'sfc:sp', 'rhos'), units='K m s-1',
          long_name='surface kinematic sensible heat flux (positive upward)', reduce='mean')  # fmt: skip
def wth(ishf, sp, rhos, ctx):
    # ERA5 fluxes are positive downward.
    return -ishf / (rhos * ctx.ifs.cpd * ctx.ifs.calc_exner(sp))


@quantity('wq', requires=('sfc:ie', 'rhos'), units='kg kg-1 m s-1',
          long_name='surface kinematic moisture flux (positive upward)', reduce='mean')  # fmt: skip
def wq(ie, rhos, ctx):
    return -ie / rhos


@quantity('z0m', requires=('sfc:fsr',), units='m', long_name='roughness length momentum', reduce='mean')
def z0m(fsr, ctx):
    return fsr


@quantity('z0h', requires=('sfc:flsr',), units='m', long_name='roughness length scalars', reduce='mean')
def z0h(flsr, ctx):
    return np.exp(flsr)


#
# Land surface.
#
@quantity('lai_low', requires=('sfc:lai_lv',), units='-', long_name='leaf area index low vegetation', reduce='mean')
def lai_low(lai_lv, ctx):
    return lai_lv


@quantity('lai_high', requires=('sfc:lai_hv',), units='-', long_name='leaf area index high vegetation', reduce='mean')
def lai_high(lai_hv, ctx):
    return lai_hv


@quantity('c_low_veg', requires=('sfc:cvl',), units='-', long_name='fraction low vegetation', reduce='mean')
def c_low_veg(cvl, ctx):
    return cvl


@quantity('c_high_veg', requires=('sfc:cvh',), units='-', long_name='fraction high vegetation', reduce='mean')
def c_high_veg(cvh, ctx):
    return cvh


@quantity('is_sea', requires=('sfc:slt',), units='-', long_name='sea mask (soil type == 0)')
def is_sea(slt, ctx):
    return np.round(slt) == 0


def _type(da, is_sea):
    return xr.where(is_sea, int(1e9), np.round(da)).astype(np.int64)


@quantity('type_soil', requires=('sfc:slt', 'is_sea'), units='-',
          long_name='ECMWF soil type (Fortran indexing!), 1e9 over sea',
          reduce=lambda da, ctx: _nearest_type(da, ctx, log=True))  # fmt: skip
def type_soil(slt, is_sea, ctx):
    return _type(slt, is_sea)


@quantity('type_low_veg', requires=('sfc:tvl', 'is_sea'), units='-',
          long_name='ECMWF low vegetation type (Fortran indexing!), 1e9 over sea', reduce=_nearest_type)  # fmt: skip
def type_low_veg(tvl, is_sea, ctx):
    return _type(tvl, is_sea)


@quantity('type_high_veg', requires=('sfc:tvh', 'is_sea'), units='-',
          long_name='ECMWF high vegetation type (Fortran indexing!), 1e9 over sea', reduce=_nearest_type)  # fmt: skip
def type_high_veg(tvh, is_sea, ctx):
    return _type(tvh, is_sea)


def _soil(*layers):
    da = xr.concat(layers, dim='soil_layer', coords='minimal', compat='override')
    da = da.assign_coords(z_soil=('soil_layer', htessel.z_soil))
    return da.transpose('time', 'soil_layer', ...)


@quantity('t_soil', requires=tuple(f'sfc:stl{i}' for i in range(1, 5)), units='K', long_name='soil temperature',
          reduce='mean')  # fmt: skip
def t_soil(stl1, stl2, stl3, stl4, ctx):
    return _soil(stl1, stl2, stl3, stl4)


@quantity('theta_soil', requires=tuple(f'sfc:swvl{i}' for i in range(1, 5)), units='m3 m-3',
          long_name='soil moisture content', reduce='mean')  # fmt: skip
def theta_soil(swvl1, swvl2, swvl3, swvl4, ctx):
    return _soil(swvl1, swvl2, swvl3, swvl4)


def _root_frac(veg_type, is_sea):
    # Vegetation types are time invariant: use the first time step.
    vt = np.maximum(np.round(veg_type.isel(time=0).values).astype(int), 1)
    rf = np.where(is_sea.isel(time=0).values, -1.0, htessel.root_fraction(vt))
    da = _like(rf, ('soil_layer', 'latitude', 'longitude'), veg_type)
    return da.assign_coords(z_soil=('soil_layer', htessel.z_soil))


@quantity('root_frac_low_veg', requires=('sfc:tvl', 'is_sea'), units='-',
          long_name='root fraction low vegetation (-1 over sea)', reduce='nearest')  # fmt: skip
def root_frac_low_veg(tvl, is_sea, ctx):
    return _root_frac(tvl, is_sea)


@quantity('root_frac_high_veg', requires=('sfc:tvh', 'is_sea'), units='-',
          long_name='root fraction high vegetation (-1 over sea)', reduce='nearest')  # fmt: skip
def root_frac_high_veg(tvh, is_sea, ctx):
    return _root_frac(tvh, is_sea)
