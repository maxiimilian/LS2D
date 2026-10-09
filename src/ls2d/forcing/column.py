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
Column quantities: large-scale forcings computed from the averaging sub-domain.

The required field quantities are passed on the sub-domain including a halo, so
`ctx.ddx()`/`ctx.ddy()` can be used. `ctx.mean()` averages over the sub-domain
without the halo, and must be used to return a column `(time, [level])`.
"""

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
from ls2d.forcing.registry import quantity
from ls2d.forcing.vertical import interp_extrap


def pressure_to_model_levels(da, p):
    """
    Interpolate column `da(time, pressure_level)` to column model level pressures `p(time, level)`.
    Extrapolates, in case ps > 1000 hPa.
    """
    return xr.apply_ufunc(
        interp_extrap, p, da.pressure_level, da,
        input_core_dims=[['level'], ['pressure_level'], ['pressure_level']],
        output_core_dims=[['level']], vectorize=True,
    )  # fmt: skip


#
# Advective tendencies.
#
@quantity('dtthl_advec', requires=('thl', 'u', 'v'), stage='column', units='K s-1',
          long_name='advective tendency liquid water potential temperature')  # fmt: skip
def dtthl_advec(thl, u, v, ctx):
    return ctx.advec(thl, u, v)


@quantity('dtqt_advec', requires=('qt', 'u', 'v'), stage='column', units='kg kg-1 s-1',
          long_name='advective tendency total specific humidity')  # fmt: skip
def dtqt_advec(qt, u, v, ctx):
    return ctx.advec(qt, u, v)


@quantity('dtu_advec', requires=('u', 'v'), stage='column', units='m s-2', long_name='advective tendency zonal wind')
def dtu_advec(u, v, ctx):
    return ctx.advec(u, u, v)


@quantity('dtv_advec', requires=('u', 'v'), stage='column', units='m s-2',
          long_name='advective tendency meridional wind')  # fmt: skip
def dtv_advec(u, v, ctx):
    return ctx.advec(v, u, v)


#
# Geostrophic wind, from the geopotential gradient on pressure levels, interpolated to the mean model level pressure.
#
@quantity('ug', requires=('phi_p', 'p'), stage='column', units='m s-1', long_name='geostrophic wind zonal component')
def ug(phi_p, p, ctx):
    return pressure_to_model_levels(ctx.mean(-ctx.ddy(phi_p) / ctx.fc), ctx.mean(p))


@quantity('vg', requires=('phi_p', 'p'), stage='column', units='m s-1',
          long_name='geostrophic wind meridional component')  # fmt: skip
def vg(phi_p, p, ctx):
    return pressure_to_model_levels(ctx.mean(ctx.ddx(phi_p) / ctx.fc), ctx.mean(p))


@quantity('dtu_coriolis', requires=('v', 'vg'), stage='column', units='m s-2',
          long_name='Coriolis tendency zonal wind, fc (v - vg)')  # fmt: skip
def dtu_coriolis(v, vg, ctx):
    return ctx.fc * (ctx.mean(v) - vg)


@quantity('dtv_coriolis', requires=('u', 'ug'), stage='column', units='m s-2',
          long_name='Coriolis tendency meridional wind, -fc (u - ug)')  # fmt: skip
def dtv_coriolis(u, ug, ctx):
    return -ctx.fc * (ctx.mean(u) - ug)


@quantity('dtu_total', requires=('dtu_advec', 'dtu_coriolis'), stage='column', units='m s-2',
          long_name='total large-scale tendency zonal wind')  # fmt: skip
def dtu_total(dtu_advec, dtu_coriolis, ctx):
    return dtu_advec + dtu_coriolis


@quantity('dtv_total', requires=('dtv_advec', 'dtv_coriolis'), stage='column', units='m s-2',
          long_name='total large-scale tendency meridional wind')  # fmt: skip
def dtv_total(dtv_advec, dtv_coriolis, ctx):
    return dtv_advec + dtv_coriolis


#
# Radiation.
#
@quantity('Th', requires=('T', 'z', 'zh'), stage='column', units='K', long_name='half level temperature')
def Th(T, z, zh, ctx):
    """
    Half level temperature, interpolated in between, and extrapolated at the surface and top.
    """
    T, z, zh = ctx.mean(T), ctx.mean(z), ctx.mean(zh)
    t, zf, zhv = T.values, z.values, zh.values

    th = np.empty_like(zhv)
    th[:, 1:-1] = 0.5 * (t[:, 1:] + t[:, :-1])
    th[:, 0] = t[:, 0] - (th[:, 1] - t[:, 0]) / (zhv[:, 1] - zf[:, 0]) * zf[:, 0]
    th[:, -1] = t[:, -1] + (t[:, -1] - th[:, -2]) / (zf[:, -1] - zhv[:, -2]) * (zhv[:, -1] - zf[:, -1])

    return xr.DataArray(th, dims=zh.dims, coords=zh.coords)
