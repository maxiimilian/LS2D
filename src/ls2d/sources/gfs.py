#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Authors: Maximilian Pierzyna (TU Delft), Claude (Anthropic)
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
NOAA GFS (0.25 degree, `pgrb2.0p25`): raw field catalogue, and the recipes that
turn the raw fields into the standard quantities (`ls2d.forcing.standard`).

GFS specifics handled here:
    - The atmosphere is only available on 41 pressure levels (0.01-1000 hPa), and is
      interpolated to terrain-following levels `p = sigma * ps`, ignoring levels below the
      surface, with the 2 m temperature/humidity and 10 m wind as values at the surface.
    - Condensate (cloud water, ice, rain, snow, graupel) is only available at 50-1000 hPa,
      and is set to zero above.
    - Surface fluxes are positive upward (W m-2). In the GRIB files they are averages since
      the start of a 6 h (3 h after +120 h) window; the reader converts them to averages
      over the last output interval, valid at the end of the interval.
    - Land surface: Noah soil layers (0-0.1, 0.1-0.4, 0.4-1, 1-2 m), no HTESSEL types.

Settings: `gfs_path` (storage), `gfs_cycle` (initialisation time, default: the last
cycle at or before `start_date`), `gfs_step` (output interval up to +120 h, default 1 h),
and `data_source` ('AWS' or 'GCS' mirror, default 'AWS').
"""

# Python modules
from dataclasses import dataclass
from typing import Optional

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
import ls2d.forcing.constants as c
from ls2d.forcing.registry import Field
from ls2d.forcing.source import Source, register_source
from ls2d.forcing.vertical import to_terrain_following

# Pressure levels of the `pgrb2.0p25` files (hPa).
# fmt: off
pressure_levels = np.array([
    0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000,
])
# fmt: on

# Terrain-following levels (fraction of surface pressure), from surface to top. Following the pressure
# levels, so that the vertical resolution is similar to that of the GFS data.
sigma = np.sort(pressure_levels / 1000.0)[::-1]

# Depth Noah soil layers (m), centre of the layers, top to bottom.
z_soil = np.array([-0.05, -0.25, -0.7, -1.5])


@dataclass(frozen=True)
class GfsField(Field):
    """
    One raw GFS variable.

    Arguments:
        name : name in LS2D (and in the NetCDF files written by `ls2d.download_gfs()`).
        levtype : 'pl' (all pressure levels) or 'sfc' (one level).
        grib : variable name in the GRIB `.idx` inventory (e.g. 'TMP').
        level : level in the `.idx` inventory (e.g. 'surface', '2 m above ground'), only for 'sfc'.
        avg : time averaged field (e.g. surface fluxes).
        fill : value for pressure levels where the field is not available.
    """

    name: str
    levtype: str
    grib: str
    level: Optional[str] = None
    avg: bool = False
    fill: Optional[float] = None
    units: str = ''
    long_name: str = ''


def _download(settings, outputs=None, fields=None, **kwargs):
    from ls2d.noaa.download_gfs import download_gfs

    return download_gfs(settings, outputs=outputs, fields=fields, **kwargs)


def _read(settings, outputs=None, fields=None):
    from ls2d.noaa.read_gfs import read_gfs

    return read_gfs(settings, outputs=outputs, fields=fields)


gfs = register_source(
    Source('gfs', 'NOAA GFS 0.25 degree forecasts (AWS or Google Cloud mirror)', GfsField, _download, _read)
)
field = gfs.field
quantity = gfs.quantity

#
# Raw field catalogue.
#
# fmt: off
# Pressure levels.
field('t',     'pl', 'TMP',  units='K',       long_name='temperature')
field('q',     'pl', 'SPFH', units='kg kg-1', long_name='specific humidity')
field('clwmr', 'pl', 'CLMR', units='kg kg-1', long_name='cloud mixing ratio', fill=0.0)
field('icmr',  'pl', 'ICMR', units='kg kg-1', long_name='ice water mixing ratio', fill=0.0)
field('rwmr',  'pl', 'RWMR', units='kg kg-1', long_name='rain mixing ratio', fill=0.0)
field('snmr',  'pl', 'SNMR', units='kg kg-1', long_name='snow mixing ratio', fill=0.0)
field('grle',  'pl', 'GRLE', units='kg kg-1', long_name='graupel', fill=0.0)
field('u',     'pl', 'UGRD', units='m s-1',   long_name='u component of wind')
field('v',     'pl', 'VGRD', units='m s-1',   long_name='v component of wind')
field('w',     'pl', 'VVEL', units='Pa s-1',  long_name='vertical velocity (pressure)')
field('gh',    'pl', 'HGT',  units='gpm',     long_name='geopotential height')
field('o3mr',  'pl', 'O3MR', units='kg kg-1', long_name='ozone mixing ratio')

# Surface / near-surface.
field('sp',    'sfc', 'PRES',  'surface',           units='Pa',      long_name='surface pressure')
field('skt',   'sfc', 'TMP',   'surface',           units='K',       long_name='surface (skin) temperature')
field('t2m',   'sfc', 'TMP',   '2 m above ground',  units='K',       long_name='2 m temperature')
field('q2m',   'sfc', 'SPFH',  '2 m above ground',  units='kg kg-1', long_name='2 m specific humidity')
field('u10',   'sfc', 'UGRD',  '10 m above ground', units='m s-1',   long_name='10 m u component of wind')
field('v10',   'sfc', 'VGRD',  '10 m above ground', units='m s-1',   long_name='10 m v component of wind')
field('shtfl', 'sfc', 'SHTFL', 'surface', avg=True, units='W m-2',   long_name='sensible heat net flux (positive upward)')
field('lhtfl', 'sfc', 'LHTFL', 'surface', avg=True, units='W m-2',   long_name='latent heat net flux (positive upward)')
field('sfcr',  'sfc', 'SFCR',  'surface',           units='m',       long_name='surface roughness')
field('land',  'sfc', 'LAND',  'surface',           units='-',       long_name='land cover (1 = land, 0 = sea)')
for _i, _layer in enumerate(('0-0.1', '0.1-0.4', '0.4-1', '1-2'), start=1):
    field(f'tsoil{_i}', 'sfc', 'TSOIL', f'{_layer} m below ground', units='K', long_name=f'soil temperature {_layer} m')
    field(f'soilw{_i}', 'sfc', 'SOILW', f'{_layer} m below ground', units='m3 m-3', long_name=f'volumetric soil moisture {_layer} m')
# fmt: on


def _tf(da, ps, surface=None, min_value=None):
    return to_terrain_following(da, ps, sigma, surface=surface, min_value=min_value)


#
# Atmosphere on terrain-following levels. Derived quantities (thl, qt, z, ph, ...) come from the core.
#
@quantity('p', requires=('ps',))
def p(ps, ctx):
    return (xr.DataArray(sigma, dims='level') * ps).transpose('time', 'level', ...)


@quantity('T', requires=('pl:t', 'ps', 'sfc:t2m'))
def T(t, ps, t2m, ctx):
    return _tf(t, ps, surface=t2m)


@quantity('qv', requires=('pl:q', 'ps', 'sfc:q2m'))
def qv(q, ps, q2m, ctx):
    return _tf(q, ps, surface=q2m, min_value=0.0)


@quantity('ql', requires=('pl:clwmr', 'pl:icmr', 'pl:rwmr', 'pl:snmr', 'pl:grle', 'ps'))
def ql(clwmr, icmr, rwmr, snmr, grle, ps, ctx):
    return _tf(clwmr + icmr + rwmr + snmr + grle, ps, min_value=0.0)


@quantity('u', requires=('pl:u', 'ps', 'sfc:u10'))
def u(u, ps, u10, ctx):
    return _tf(u, ps, surface=u10)


@quantity('v', requires=('pl:v', 'ps', 'sfc:v10'))
def v(v, ps, v10, ctx):
    return _tf(v, ps, surface=v10)


@quantity('omega', requires=('pl:w', 'ps'))
def omega(w, ps, ctx):
    return _tf(w, ps)


@quantity('o3', requires=('pl:o3mr', 'ps'))
def o3(o3mr, ps, ctx):
    return _tf(o3mr, ps, min_value=0.0) * c.md_mo3


@quantity('phi_p', requires=('pl:gh',))
def phi_p(gh, ctx):
    # Geopotential on the pressure levels themselves (below ground: extrapolated by NCEP).
    return gh * c.grav


#
# Surface. GFS fluxes are positive upward.
#
@quantity('ps', requires=('sfc:sp',))
def ps(sp, ctx):
    return sp


@quantity('ts', requires=('sfc:skt',))
def ts(skt, ctx):
    return skt


@quantity('sst', requires=('sfc:skt', 'sfc:land'))
def sst(skt, land, ctx):
    # GFS surface temperature over water = SST; undefined (NaN) over land, as in ERA5.
    return skt.where(land < 0.5)


@quantity('wth', requires=('sfc:shtfl', 'ps', 'rhos'))
def wth(shtfl, ps, rhos, ctx):
    return shtfl / (rhos * c.cpd * (ps / c.p0) ** (c.Rd / c.cpd))


@quantity('wq', requires=('sfc:lhtfl', 'rhos'))
def wq(lhtfl, rhos, ctx):
    return lhtfl / (rhos * c.Lv)


@quantity('z0m', requires=('sfc:sfcr',))
def z0m(sfcr, ctx):
    return sfcr


#
# Land surface (Noah). Undefined (NaN) over sea; averages are over the land points.
#
def _soil(*layers):
    da = xr.concat(layers, dim='soil_layer', coords='minimal', compat='override')
    da = da.assign_coords(z_soil=('soil_layer', z_soil))
    return da.transpose('time', 'soil_layer', ...)


@quantity('t_soil', requires=tuple(f'sfc:tsoil{i}' for i in range(1, 5)))
def t_soil(t1, t2, t3, t4, ctx):
    return _soil(t1, t2, t3, t4)


@quantity('theta_soil', requires=tuple(f'sfc:soilw{i}' for i in range(1, 5)))
def theta_soil(w1, w2, w3, w4, ctx):
    return _soil(w1, w2, w3, w4)
