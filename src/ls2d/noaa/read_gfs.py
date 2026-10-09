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
Read the GFS files downloaded by `ls2d.download_gfs()` into the raw dataset (see `ls2d.forcing.raw`).
"""

# Python modules
import os

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
import ls2d.noaa.gfs_tools as gt
from ls2d.core.logger import logger
from ls2d.forcing.raw import missing_fields, standardize
from ls2d.forcing.source import get_source
from ls2d.noaa.download_gfs import gfs_file


def interval_means(values, hours, starts):
    """
    Convert GFS averages since the start of the averaging window to averages over the
    last output interval (`hours[i-1]` to `hours[i]`), valid at `hours[i]`.

    Arguments:
        values : array (time, ...), average from `starts[i]` to `hours[i]`.
        hours : forecast hours.
        starts : start of the averaging window per time (-1 = not available, e.g. analysis).

    Returns:
        array (time, ...); NaN where the interval mean can not be calculated.
    """
    out = np.full_like(values, np.nan, dtype=float)
    for i, (h, a) in enumerate(zip(hours, starts)):
        if a < 0:
            continue
        if i > 0 and a == hours[i - 1]:
            out[i] = values[i]  # Window starts at the previous output time.
        elif i > 0 and a < hours[i - 1] and starts[i - 1] == a:
            hp = hours[i - 1]
            out[i] = ((h - a) * values[i] - (hp - a) * values[i - 1]) / (h - hp)
        elif i == 0 and a == h - 1 and h <= gt.max_hourly:
            out[i] = values[i]  # One hour window, no previous output needed.
    return out


def read_gfs(settings, outputs=None, fields=None):
    """
    Read the GFS files downloaded by `ls2d.download_gfs()`, and return the raw dataset
    for the period `start_date` to `end_date`. See `ls2d.download_gfs()` for the settings.
    """

    src = get_source('gfs')
    fields = src.required_fields(outputs) if fields is None else fields
    c = gt.cycle(settings)
    hours = gt.forecast_hours(settings)

    logger.info(f'Reading GFS cycle {c:%Y-%m-%d %H} UTC for period: {settings["start_date"]} to {settings["end_date"]}')

    avg = [f for f in fields if f.avg]
    other = [f for f in fields if not f.avg]

    def open_hour(hour, flds):
        nc_file = gfs_file(settings, c, hour)
        if not os.path.exists(nc_file):
            msg = f'File "{nc_file}" does not exist. Run `ls2d.download_gfs()` first.'
            logger.error(msg)
            raise FileNotFoundError(msg)
        ds = xr.open_dataset(nc_file)
        missing = missing_fields(ds, flds)
        if missing:
            msg = (
                f'GFS file "{nc_file}" does not contain the field(s) {[m.key for m in missing]}. '
                f'Re-download with `ls2d.download_gfs()`, which updates files with missing fields.'
            )
            logger.error(msg)
            raise KeyError(msg)
        names = [f.name for f in flds] + [f'{f.name}_avg_start' for f in flds if f.avg]
        return ds[names]

    ds = xr.concat([open_hour(h, fields) for h in hours], dim='time')

    # Time averaged fields: interval means, using the previous output hour if needed.
    if avg:
        prev = gt.previous_hour(hours[0], settings)
        avg_hours = list(hours)
        avg_ds = ds[[f.name for f in avg] + [f'{f.name}_avg_start' for f in avg]]
        if prev is not None and os.path.exists(gfs_file(settings, c, prev)):
            avg_ds = xr.concat([open_hour(prev, avg), avg_ds], dim='time')
            avg_hours = [prev] + avg_hours

        n_extra = len(avg_hours) - len(hours)
        for f in avg:
            starts = avg_ds[f'{f.name}_avg_start'].values.astype(int)
            values = interval_means(avg_ds[f.name].values, avg_hours, starts)[n_extra:]

            # Analysis (or missing previous hour): use the first available interval mean.
            nan = np.isnan(values).all(axis=tuple(range(1, values.ndim)))
            if nan.all():
                raise ValueError(f'No valid values for GFS field {f.key}; download at least two forecast hours.')
            if nan.any():
                first = np.argmin(nan)
                logger.warning(
                    f'GFS {f.key} not available at f{hours[0]:03d} (analysis or missing previous hour): '
                    f'using the interval mean of f{hours[first]:03d}.'
                )
                for i in np.where(nan)[0]:
                    j = i + np.argmin(nan[i:]) if not nan[i:].all() else first
                    values[i] = values[j]
            ds[f.name] = ds[f.name].copy(data=values)

    ds = ds[[f.name for f in other + avg]]
    attrs = dict(
        central_lat=settings['central_lat'],
        central_lon=settings['central_lon'],
        source=f'NOAA GFS 0.25 degree, cycle {c:%Y-%m-%d %H} UTC',
        ls2d_source='gfs',
        gfs_cycle=f'{c:%Y-%m-%d %H}:00',
    )
    return standardize({'pl': ds, 'sfc': ds}, fields, attrs).load()
