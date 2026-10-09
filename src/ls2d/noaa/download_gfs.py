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
Download NOAA GFS 0.25 degree forecasts from the public AWS (or Google Cloud) archive.

Only the GRIB messages of the required fields are downloaded (HTTP byte ranges, using the
`.idx` inventories), cropped to the domain, and saved as one NetCDF file per forecast hour:
`gfs_path/case_name/gfs/yyyymmddhh/gfs.fXXX.nc` (yyyymmddhh = cycle).
"""

# Python modules
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

# Third party modules
import numpy as np
import xarray as xr

# LS2D modules
import ls2d.noaa.gfs_tools as gt
from ls2d.core.logger import logger
from ls2d.forcing.raw import fields_to_download
from ls2d.forcing.source import get_source


def gfs_file(settings, c, hour):
    """
    Path of the NetCDF file of cycle `c` and forecast `hour`.
    """
    if 'gfs_path' not in settings:
        raise KeyError('Setting `gfs_path` (storage location of the GFS data) is missing')
    return os.path.join(settings['gfs_path'], settings['case_name'], 'gfs', f'{c:%Y%m%d%H}', f'gfs.f{hour:03d}.nc')


def mirror(settings):
    m = settings.get('data_source', 'AWS')
    if m not in gt.mirrors:
        raise ValueError(f'Unknown GFS data source "{m}", choose from {list(gt.mirrors)}')
    return m


def download_plan(settings, fields):
    """
    (forecast hour, fields) to download: all fields for the requested hours, plus the
    time averaged fields of the hour before, needed to convert the averages to interval means.
    """
    hours = gt.forecast_hours(settings)
    plan = [(h, list(fields)) for h in hours]
    prev = gt.previous_hour(hours[0], settings)
    avg = [f for f in fields if f.avg]
    if prev is not None and avg:
        plan.insert(0, (prev, avg))
    return plan


def _inventory(m, c, hour):
    """
    URL and parsed inventory of a GRIB file (trying the post and pre March 2021 directory layout).
    """
    for atmos in (True, False):
        url = gt.file_url(m, c, hour, atmos)
        try:
            return url, gt.parse_idx(gt.fetch(f'{url}.idx').decode())
        except FileNotFoundError:
            continue
    msg = f'GFS cycle {c:%Y-%m-%d %H} UTC, forecast hour {hour} is not available on {m} ({gt.mirrors[m]})'
    logger.error(msg)
    raise FileNotFoundError(msg)


def download_hour(settings, c, hour, fields, nc_file, max_workers=16):
    """
    Download, crop, and save `fields` of one forecast hour.
    """
    m = mirror(settings)
    url, msgs = _inventory(m, c, hour)
    selected = gt.select_messages(msgs, fields, hour)

    # Download and decode the selected messages, merged into as few range requests as possible.
    # Each worker downloads one range and decodes its messages (eccodes releases the GIL).
    def work(r):
        start, end, group = r
        blob = gt.fetch(url, start, end)
        return [gt.decode(blob[msg['start'] - start : None if msg['end'] is None else msg['end'] - start + 1])
                for msg in group]  # fmt: skip

    ranges = gt.merge_ranges([msg for _, _, msg in selected])
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        decoded = {}
        for (_, _, group), results in zip(ranges, pool.map(work, ranges)):
            for msg, result in zip(group, results):
                decoded[msg['start']] = result

    # Crop to the domain.
    lats, lons = decoded[selected[0][2]['start']][1:]
    ilat, ilon, lons_out = gt.crop_indices(
        lats, lons, settings['central_lat'], settings['central_lon'], settings['area_size']
    )
    lats_out = lats[ilat]
    data = {}
    for f, level, msg in selected:
        values, msg_lats, msg_lons = decoded[msg['start']]
        if not (np.array_equal(msg_lats, lats) and np.array_equal(msg_lons, lons)):
            raise ValueError(f'GRIB messages on different grids (f{hour:03d})')
        data[(f.name, level)] = (values[np.ix_(ilat, ilon)], msg)

    levels = np.array(sorted({lv for (_, lv) in data if lv is not None}, reverse=True))
    shape = (lats_out.size, lons_out.size)
    valid = datetime.datetime(c.year, c.month, c.day, c.hour) + datetime.timedelta(hours=hour)

    variables = {}
    for f in fields:
        attrs = dict(units=f.units, long_name=f.long_name, grib=f'{f.grib}:{f.level or "isobaric"}')
        if f.levtype == 'pl':
            arr = np.full((levels.size,) + shape, np.nan)
            for k, lv in enumerate(levels):
                if (f.name, lv) in data:
                    arr[k] = data[(f.name, lv)][0]
                elif f.fill is not None:
                    arr[k] = f.fill
                else:
                    raise KeyError(f'GFS field {f.key} not available at {lv} hPa (f{hour:03d})')
            variables[f.name] = (('time', 'pressure_level', 'latitude', 'longitude'), arr[None], attrs)
        else:
            if (f.name, None) in data:
                arr, msg = data[(f.name, None)]
                start = gt.avg_start(msg['fcst'])
            else:  # Time averaged field, not available in the analysis.
                arr, start = np.full(shape, np.nan), None
            variables[f.name] = (('time', 'latitude', 'longitude'), arr[None], attrs)
            if f.avg:
                variables[f'{f.name}_avg_start'] = (('time',), [-1 if start is None else start])

    ds = xr.Dataset(
        variables,
        coords=dict(
            time=('time', [np.datetime64(valid, 'ns')]),
            fcst_hour=('time', [hour]),
            pressure_level=('pressure_level', levels, dict(units='hPa')),
            latitude=('latitude', lats_out, dict(units='degrees_north')),
            longitude=('longitude', lons_out, dict(units='degrees_east')),
        ),
        attrs=dict(
            source=f'NOAA GFS 0.25 degree ({url})',
            cycle=f'{c:%Y-%m-%d %H}:00 UTC',
            history=f'Downloaded by (LS)2D on {datetime.datetime.now():%Y-%m-%d %H:%M}',
        ),
    )

    # Write to temporary file first, so that an interrupted download does not leave a valid looking file.
    os.makedirs(os.path.dirname(nc_file), exist_ok=True)
    tmp = f'{nc_file}.tmp'
    ds.to_netcdf(tmp)
    os.replace(tmp, nc_file)


def download_gfs(settings, outputs=None, fields=None, max_workers=16):
    """
    Download the GFS fields needed for `outputs` for the forecast between `start_date` and `end_date`.

    Arguments:
        settings : dict
            Dictionary with keys `central_lat`, `central_lon`, `area_size`, `gfs_path`, `case_name`,
            `start_date`, `end_date`, and optionally `gfs_cycle` (default: last cycle at or before
            `start_date`), `gfs_step` (output interval up to +120 h, default 1 h), and `data_source`
            ('AWS' (default) or 'GCS').
        outputs : list of str, optional
            LES outputs and/or column variables (see `ls2d.les_outputs()`). Default: all that GFS can provide.
        fields : list of `GfsField`, optional
            Download exactly these fields (overrides `outputs`).
        max_workers : int
            Number of parallel HTTP requests.
    """

    src = get_source('gfs')
    fields = src.required_fields(outputs) if fields is None else fields
    c = gt.cycle(settings)
    m = mirror(settings)

    logger.info(
        f'Downloading GFS ({m}) cycle {c:%Y-%m-%d %H} UTC for period: {settings["start_date"]} to {settings["end_date"]}'
    )

    for hour, flds in download_plan(settings, fields):
        nc_file = gfs_file(settings, c, hour)
        to_download = fields_to_download(nc_file, flds, src.registry.fields())
        if not to_download:
            logger.debug(f'Found {nc_file} local')
            continue
        if os.path.isfile(nc_file):
            logger.info(f'{nc_file} misses required field(s), downloading again')
        logger.info(f'Downloading GFS f{hour:03d} ({len(to_download)} fields)')
        download_hour(settings, c, hour, to_download, nc_file, max_workers)

    return True
