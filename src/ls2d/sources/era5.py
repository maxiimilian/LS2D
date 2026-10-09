#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
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
ERA5: raw field catalogue (with the names used by CDS, MARS, and Google ARCO), and the
recipes that turn the raw fields into the standard quantities (`ls2d.forcing.standard`).

Everything ERA5/IFS specific lives here: the L137 hybrid levels, the sign
conventions of the surface fluxes, and the HTESSEL land surface.
"""

# Python modules
from dataclasses import dataclass
from typing import Optional

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
import ls2d.ecmwf.htessel as htessel
import ls2d.forcing.constants as c
from ls2d.ecmwf.IFS_tools import IFS_tools
from ls2d.forcing.registry import Field
from ls2d.forcing.source import Source, register_source
from ls2d.forcing.standard import d3h, d3s_static

ifs = IFS_tools('L137')


@dataclass(frozen=True)
class Era5Field(Field):
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


def _download(settings, outputs=None, fields=None, **kwargs):
    from ls2d.ecmwf.download_era5 import download_era5

    return download_era5(settings, outputs=outputs, fields=fields, **kwargs)


def _read(settings, outputs=None, fields=None):
    from ls2d.ecmwf.read_era5 import read_era5

    return read_era5(settings, outputs=outputs, fields=fields)


era5 = register_source(
    Source(
        'era5',
        'ERA5 reanalysis (CDS, MARS, or Google ARCO; `data_source` setting)',
        field_class=Era5Field,
        download=_download,
        read=_read,
    )
)
field = era5.field
quantity = era5.quantity

#
# Raw field catalogue.
#
# fmt: off
# Model level analysis. `param` as used by the CDS "complete" and MARS requests.
field('t',    'ml', '130', arco='temperature',                         units='K',       long_name='temperature')
field('u',    'ml', '131', arco='u_component_of_wind',                 units='m s-1',   long_name='u component of wind')
field('v',    'ml', '132', arco='v_component_of_wind',                 units='m s-1',   long_name='v component of wind')
field('w',    'ml', '135', arco='vertical_velocity',                   units='Pa s-1',  long_name='vertical velocity')
field('q',    'ml', '133', arco='specific_humidity',                   units='kg kg-1', long_name='specific humidity')
field('clwc', 'ml', '246', arco='specific_cloud_liquid_water_content', units='kg kg-1', long_name='specific cloud liquid water content')
field('ciwc', 'ml', '247', arco='specific_cloud_ice_water_content',    units='kg kg-1', long_name='specific cloud ice water content')
field('crwc', 'ml', '75',  arco='specific_rain_water_content',         units='kg kg-1', long_name='specific rain water content')
field('cswc', 'ml', '76',  arco='specific_snow_water_content',         units='kg kg-1', long_name='specific snow water content')
field('o3',   'ml', '203', arco='ozone_mass_mixing_ratio',             units='kg kg-1', long_name='ozone mass mixing ratio')

# Pressure level analysis.
field('z', 'pl', '129.128', cds='geopotential', arco='geopotential', units='m2 s-2', long_name='geopotential')

# Surface / single level analysis.
field('sp',     'sfc', '134.128', cds='surface_pressure',                                 arco='surface_pressure',                                 units='Pa',          long_name='surface pressure')
field('skt',    'sfc', '235.128', cds='skin_temperature',                                 arco='skin_temperature',                                 units='K',           long_name='skin temperature')
field('sst',    'sfc', '34.128',  cds='sea_surface_temperature',                          arco='sea_surface_temperature',                          units='K',           long_name='sea surface temperature')
field('ishf',   'sfc', '231.128', cds='instantaneous_surface_sensible_heat_flux',         arco='instantaneous_surface_sensible_heat_flux',         units='W m-2',       long_name='instantaneous surface sensible heat flux')
field('ie',     'sfc', '232.128', cds='instantaneous_moisture_flux',                      arco='instantaneous_moisture_flux',                      units='kg m-2 s-1',  long_name='instantaneous moisture flux')
field('fsr',    'sfc', '244.128', cds='forecast_surface_roughness',                       arco='forecast_surface_roughness',                       units='m',           long_name='forecast surface roughness')
field('flsr',   'sfc', '245.128', cds='forecast_logarithm_of_surface_roughness_for_heat', arco='forecast_logarithm_of_surface_roughness_for_heat', units='~',           long_name='forecast logarithm of surface roughness for heat')
field('slt',    'sfc', '43.128',  cds='soil_type',                                        arco='soil_type',                                        units='~',           long_name='soil type')
field('tvl',    'sfc', '29.128',  cds='type_of_low_vegetation',                           arco='type_of_low_vegetation',                           units='~',           long_name='type of low vegetation')
field('tvh',    'sfc', '30.128',  cds='type_of_high_vegetation',                          arco='type_of_high_vegetation',                          units='~',           long_name='type of high vegetation')
field('lai_lv', 'sfc', '66.128',  cds='leaf_area_index_low_vegetation',                   arco='leaf_area_index_low_vegetation',                   units='m2 m-2',      long_name='leaf area index, low vegetation')
field('lai_hv', 'sfc', '67.128',  cds='leaf_area_index_high_vegetation',                  arco='leaf_area_index_high_vegetation',                  units='m2 m-2',      long_name='leaf area index, high vegetation')
field('cvl',    'sfc', '27.128',  cds='low_vegetation_cover',                             arco='low_vegetation_cover',                             units='(0 - 1)',     long_name='low vegetation cover')
field('cvh',    'sfc', '28.128',  cds='high_vegetation_cover',                            arco='high_vegetation_cover',                            units='(0 - 1)',     long_name='high vegetation cover')
for _i, _param in enumerate(('139.128', '170.128', '183.128', '236.128'), start=1):
    field(f'stl{_i}', 'sfc', _param, cds=f'soil_temperature_level_{_i}', arco=f'soil_temperature_level_{_i}', units='K', long_name=f'soil temperature level {_i}')
for _i, _param in enumerate(('39.128', '40.128', '41.128', '42.128'), start=1):
    field(f'swvl{_i}', 'sfc', _param, cds=f'volumetric_soil_water_layer_{_i}', arco=f'volumetric_soil_water_layer_{_i}', units='m3 m-3', long_name=f'volumetric soil water layer {_i}')
# fmt: on


def _like(values, dims, *templates):
    """
    DataArray from numpy `values`, with the coordinates of `templates` for `dims`.
    """
    coords = {}
    for t in templates:
        for name, crd in t.coords.items():
            if set(crd.dims) <= set(dims) and name not in coords:
                coords[name] = crd
    return xr.DataArray(values, dims=dims, coords=coords)


#
# Atmosphere on L137 model levels. The derived quantities (thl, qt, z, ...) come from `ls2d.forcing.derived`.
#
@quantity('T', requires=('ml:t',))
def T(t, ctx):
    return t


@quantity('qv', requires=('ml:q',))
def qv(q, ctx):
    return q


@quantity('ql', requires=('ml:clwc', 'ml:ciwc', 'ml:crwc', 'ml:cswc'))
def ql(clwc, ciwc, crwc, cswc, ctx):
    return clwc + ciwc + crwc + cswc


@quantity('u', requires=('ml:u',))
def u(u, ctx):
    return u


@quantity('v', requires=('ml:v',))
def v(v, ctx):
    return v


@quantity('omega', requires=('ml:w',))
def omega(w, ctx):
    return w


@quantity('ph', requires=('sfc:sp',))
def ph(sp, ctx):
    """
    Half level pressure from the L137 hybrid coefficients.
    """
    p = ifs.a[None, :, None, None] + ifs.b[None, :, None, None] * sp.values[:, None]
    p[:, -1] = 0.34  # As in `IFS_tools.calc_half_level_pressure()`
    return _like(p, d3h, sp)


@quantity('o3', requires=('ml:o3',))
def o3(o3, ctx):
    return o3 * c.md_mo3


@quantity('phi_p', requires=('pl:z',))
def phi_p(z, ctx):
    return z


#
# Surface. ERA5 fluxes are positive downward.
#
@quantity('ps', requires=('sfc:sp',))
def ps(sp, ctx):
    return sp


@quantity('ts', requires=('sfc:skt',))
def ts(skt, ctx):
    return skt


@quantity('sst', requires=('sfc:sst',))
def sst(sst, ctx):
    return sst


@quantity('wth', requires=('sfc:ishf', 'ps', 'rhos'))
def wth(ishf, ps, rhos, ctx):
    return -ishf / (rhos * c.cpd * (ps / c.p0) ** (c.Rd / c.cpd))


@quantity('wq', requires=('sfc:ie', 'rhos'))
def wq(ie, rhos, ctx):
    return -ie / rhos


@quantity('z0m', requires=('sfc:fsr',))
def z0m(fsr, ctx):
    return fsr


@quantity('z0h', requires=('sfc:flsr',))
def z0h(flsr, ctx):
    return np.exp(flsr)


#
# Land surface (HTESSEL).
#
@quantity('lai_low', requires=('sfc:lai_lv',))
def lai_low(lai_lv, ctx):
    return lai_lv


@quantity('lai_high', requires=('sfc:lai_hv',))
def lai_high(lai_hv, ctx):
    return lai_hv


@quantity('c_low_veg', requires=('sfc:cvl',))
def c_low_veg(cvl, ctx):
    return cvl


@quantity('c_high_veg', requires=('sfc:cvh',))
def c_high_veg(cvh, ctx):
    return cvh


@quantity('is_sea', requires=('sfc:slt',), units='-', long_name='sea mask (soil type == 0)')
def is_sea(slt, ctx):
    return np.round(slt) == 0


def _type(da, is_sea):
    return xr.where(is_sea, int(1e9), np.round(da)).astype(np.int64)


@quantity('type_soil', requires=('sfc:slt', 'is_sea'))
def type_soil(slt, is_sea, ctx):
    return _type(slt, is_sea)


@quantity('type_low_veg', requires=('sfc:tvl', 'is_sea'))
def type_low_veg(tvl, is_sea, ctx):
    return _type(tvl, is_sea)


@quantity('type_high_veg', requires=('sfc:tvh', 'is_sea'))
def type_high_veg(tvh, is_sea, ctx):
    return _type(tvh, is_sea)


def _soil(*layers):
    da = xr.concat(layers, dim='soil_layer', coords='minimal', compat='override')
    da = da.assign_coords(z_soil=('soil_layer', htessel.z_soil))
    return da.transpose('time', 'soil_layer', ...)


@quantity('t_soil', requires=tuple(f'sfc:stl{i}' for i in range(1, 5)))
def t_soil(stl1, stl2, stl3, stl4, ctx):
    return _soil(stl1, stl2, stl3, stl4)


@quantity('theta_soil', requires=tuple(f'sfc:swvl{i}' for i in range(1, 5)))
def theta_soil(swvl1, swvl2, swvl3, swvl4, ctx):
    return _soil(swvl1, swvl2, swvl3, swvl4)


def _root_frac(veg_type, is_sea):
    # Vegetation types are time invariant: use the first time step.
    vt = np.maximum(np.round(veg_type.isel(time=0).values).astype(int), 1)
    rf = np.where(is_sea.isel(time=0).values, -1.0, htessel.root_fraction(vt))
    da = _like(rf, d3s_static, veg_type)
    return da.assign_coords(z_soil=('soil_layer', htessel.z_soil))


@quantity('root_frac_low_veg', requires=('sfc:tvl', 'is_sea'))
def root_frac_low_veg(tvl, is_sea, ctx):
    return _root_frac(tvl, is_sea)


@quantity('root_frac_high_veg', requires=('sfc:tvh', 'is_sea'))
def root_frac_high_veg(tvh, is_sea, ctx):
    return _root_frac(tvh, is_sea)
