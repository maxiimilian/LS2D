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
Vertical helpers for data sources, e.g. to convert pressure level data
(with levels below the surface) to a terrain-following vertical coordinate.
"""

# Third party modules
import numpy as np
import xarray as xr


def interp_extrap(x, xp, fp):
    """
    Linear interpolation, with linear extrapolation outside `xp`.
    """
    i = np.argsort(xp)
    xp, fp = xp[i], fp[i]
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def mid_levels(da, dim_in, dim_out):
    """
    Values in between the levels of `da` along `dim_in` (n -> n-1 levels, as `dim_out`).
    """
    lower = da.isel({dim_in: slice(None, -1)}).drop_vars(dim_in, errors='ignore')
    upper = da.isel({dim_in: slice(1, None)}).drop_vars(dim_in, errors='ignore')
    return (0.5 * (lower + upper)).rename({dim_in: dim_out})


def half_levels_from_full(p, ps, p_top_min=0.34):
    """
    Half level pressure from full level pressure `p(time, level, ...)` and surface pressure `ps`:
    surface pressure at the first half level, values in between the full levels, and
    extrapolation (limited to `p_top_min`) at the top.
    """
    pv = p.transpose('time', 'level', ...).values
    ph = np.empty((pv.shape[0], pv.shape[1] + 1) + pv.shape[2:])
    ph[:, 0] = ps.transpose('time', ...).values
    ph[:, 1:-1] = 0.5 * (pv[:, 1:] + pv[:, :-1])
    ph[:, -1] = np.maximum(pv[:, -1] - 0.5 * (pv[:, -2] - pv[:, -1]), p_top_min)

    coords = {k: c for k, c in p.coords.items() if 'level' not in c.dims}
    dims = ('time', 'level_half') + tuple(d for d in p.dims if d not in ('time', 'level'))
    return xr.DataArray(ph, dims=dims, coords=coords)


def _to_sigma_column(p_target, p_levels, values, ps, surface=np.nan):
    """
    Interpolate one column in ln(p) from the pressure levels above the surface to `p_target`.
    If `surface` is given (finite), it is used as value at `p = ps`. Otherwise, values below the
    lowest pressure level above the surface are extrapolated linearly.
    """
    valid = (p_levels < ps) & np.isfinite(values)
    xp, fp = p_levels[valid], values[valid]
    if np.isfinite(surface):
        xp, fp = np.append(xp, ps), np.append(fp, surface)
    if xp.size < 2:
        raise ValueError(f'Less than two pressure levels above the surface (ps = {ps:.0f} Pa)')
    return interp_extrap(np.log(p_target), np.log(xp), fp)


def to_terrain_following(da, ps, sigma, surface=None, min_value=None):
    """
    Convert pressure level data to terrain-following levels `p = sigma * ps`.

    Pressure levels below the surface (`pressure_level >= ps`) are ignored. Between the lowest
    pressure level above the surface and the surface, values are interpolated (in ln(p)) to the
    `surface` value at `p = ps` (e.g. the 2 m temperature) if given, otherwise extrapolated linearly.

    Arguments:
        da : xarray.DataArray
            Data with dims (time, pressure_level, latitude, longitude), `pressure_level` in Pa.
            NaN values (e.g. levels where a variable is not available) are ignored.
        ps : xarray.DataArray
            Surface pressure (time, latitude, longitude) in Pa.
        sigma : array
            Target levels as fraction of the surface pressure, from surface (1) to top.
        surface : xarray.DataArray, optional
            Values at the surface (time, latitude, longitude).
        min_value : float, optional
            Lower limit, e.g. 0 for mixing ratios (extrapolation can give negative values).

    Returns:
        xarray.DataArray with dims (time, level, latitude, longitude).
    """
    sigma = xr.DataArray(np.asarray(sigma, dtype=float), dims='level')
    p_target = sigma * ps
    if surface is None:
        surface = xr.full_like(ps, np.nan, dtype=float)
    out = xr.apply_ufunc(
        _to_sigma_column, p_target, da.pressure_level, da, ps, surface,
        input_core_dims=[['level'], ['pressure_level'], ['pressure_level'], [], []],
        output_core_dims=[['level']], vectorize=True,
    )  # fmt: skip
    if min_value is not None:
        out = out.clip(min=min_value)
    return out.transpose('time', 'level', 'latitude', 'longitude')
