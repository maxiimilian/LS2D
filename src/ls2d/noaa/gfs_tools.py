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
Helpers for the NOAA GFS 0.25 degree GRIB2 files (`pgrb2.0p25`): forecast hours,
`.idx` inventories, byte range selection, and GRIB decoding.
"""

# Python modules
import datetime
import re
import time

# Third party modules
import numpy as np
import requests

# LS2D modules
from ls2d.core.logger import logger

mirrors = {
    'AWS': 'https://noaa-gfs-bdp-pds.s3.amazonaws.com',
    'GCS': 'https://storage.googleapis.com/global-forecast-system',
}

# Hourly output up to +120 h, 3-hourly up to +384 h.
max_hourly = 120
max_hour = 384


def cycle(settings):
    """
    GFS cycle (initialisation time): `settings['gfs_cycle']`, or the last
    00/06/12/18 UTC cycle at or before `settings['start_date']`.
    """
    if settings.get('gfs_cycle') is not None:
        c = settings['gfs_cycle']
        if c.hour % 6 != 0 or c.minute != 0 or c.second != 0:
            raise ValueError(f'GFS cycle should be 00, 06, 12, or 18 UTC, not {c}')
        return datetime.datetime(c.year, c.month, c.day, c.hour)
    s = settings['start_date']
    return datetime.datetime(s.year, s.month, s.day, s.hour - s.hour % 6)


def forecast_hours(settings):
    """
    Forecast hours (relative to `cycle(settings)`) covering `start_date` to `end_date`.
    Hourly up to +120 h (or every `settings['gfs_step']` hours), 3-hourly after.
    """
    c = cycle(settings)
    step = settings.get('gfs_step', 1)
    h0 = (settings['start_date'] - c).total_seconds() / 3600
    h1 = (settings['end_date'] - c).total_seconds() / 3600

    if h0 < 0:
        raise ValueError(f'start_date {settings["start_date"]} is before the GFS cycle {c}')
    if h1 > max_hour:
        raise ValueError(f'end_date {settings["end_date"]} is beyond the last GFS forecast hour (+{max_hour} h)')

    hours = [h for h in range(0, max_hourly + 1, step)] + list(range(max_hourly + 3, max_hour + 1, 3))
    hours = sorted(set(hours))
    selected = [h for h in hours if h0 <= h <= h1]
    if h0 not in hours or h1 not in hours:
        raise ValueError(
            f'start_date/end_date (+{h0:g} h, +{h1:g} h from {c:%Y-%m-%d %H} UTC) should be GFS output times '
            f'(every {step} h up to +{max_hourly} h, every 3 h after).'
        )
    if len(selected) < 2:
        raise ValueError('At least two GFS output times are needed.')
    return selected


def previous_hour(hour, settings):
    """
    Output hour before `hour` (None for the analysis).
    """
    if hour == 0:
        return None
    return hour - (3 if hour > max_hourly else settings.get('gfs_step', 1))


def file_url(mirror, c, hour, atmos=True):
    """
    URL of the GRIB2 file of cycle `c` and forecast `hour`. Since March 2021, files are in an `atmos` sub directory.
    """
    sub = '/atmos' if atmos else ''
    return f'{mirrors[mirror]}/gfs.{c:%Y%m%d}/{c:%H}{sub}/gfs.t{c:%H}z.pgrb2.0p25.f{hour:03d}'


#
# Inventory (.idx).
#
_avg = re.compile(r'(\d+)-(\d+) hour ave fcst')


def parse_idx(text):
    """
    Parse a GRIB `.idx` inventory. Returns list of dicts with keys
    `var`, `level`, `fcst`, `start`, `end` (byte range, `end` inclusive; None for the last message).
    """
    msgs = []
    for line in text.strip().splitlines():
        parts = line.split(':')
        msgs.append(dict(var=parts[3], level=parts[4], fcst=parts[5], start=int(parts[1])))
    for m, nxt in zip(msgs, msgs[1:] + [None]):
        m['end'] = nxt['start'] - 1 if nxt is not None else None
    return msgs


def avg_start(fcst):
    """
    Start hour of the averaging period of a time averaged field ('0-6 hour ave fcst' -> 0), else None.
    """
    m = _avg.search(fcst)
    return int(m.group(1)) if m else None


def select_messages(msgs, fields, hour):
    """
    Select the GRIB messages for `fields` from the inventory `msgs`.

    Returns:
        list of (field, pressure level in hPa or None, message dict).
    Raises:
        KeyError if a field is not available (except time averaged fields in the analysis).
    """
    out = []
    for f in fields:
        if f.levtype == 'pl':
            found = [(float(m['level'][:-3]), m) for m in msgs if m['var'] == f.grib and m['level'].endswith(' mb')]
        else:
            found = [(None, m) for m in msgs if m['var'] == f.grib and m['level'] == f.level]
            if f.avg:
                found = [(lv, m) for lv, m in found if avg_start(m['fcst']) is not None]
        if not found:
            if f.avg and hour == 0:
                continue  # No time averaged fields in the analysis.
            raise KeyError(f'GFS field {f.key} ({f.grib}:{f.level or "isobaric"}) not found in f{hour:03d}')
        out += [(f, lv, m) for lv, m in found]
    return out


def merge_ranges(msgs, max_gap=0):
    """
    Merge the byte ranges of (sorted) messages into as few HTTP range requests as possible.
    Returns list of (start, end, [messages]).
    """
    out = []
    for m in sorted(msgs, key=lambda m: m['start']):
        if out and out[-1][1] is not None and m['start'] <= out[-1][1] + 1 + max_gap:
            start, _, group = out[-1]
            out[-1] = (start, m['end'], group + [m])
        else:
            out.append((m['start'], m['end'], [m]))
    return out


def fetch(url, start=None, end=None, retries=4, timeout=120):
    """
    HTTP GET (optionally a byte range), with retries and exponential backoff.
    """
    headers = {}
    if start is not None:
        headers['Range'] = f'bytes={start}-{"" if end is None else end}'
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            if i == retries:
                raise
            wait = 2 ** (i + 1)
            logger.warning(f'Download of {url} failed ({e}), retrying in {wait} s')
            time.sleep(wait)


#
# Grid and decoding.
#
def crop_indices(lats, lons, central_lat, central_lon, area_size):
    """
    Indices of the box around the grid point nearest to (`central_lat`, `central_lon`),
    with +/- `area_size` degrees plus one extra grid point for the gradients. Handles
    crossing of the 0 degree meridian. Returns (latitude indices, longitude indices, output longitudes).
    """
    resolution = abs(float(lats[1] - lats[0]))
    n = int(round(area_size / resolution)) + 1
    jc = int(np.abs(lats - central_lat).argmin())
    if jc - n < 0 or jc + n > lats.size - 1:
        raise ValueError('Domain too close to the poles')
    ilat = np.arange(jc - n, jc + n + 1)

    dlon = ((lons - central_lon % 360 + 180) % 360) - 180
    ic = int(np.abs(dlon).argmin())
    ilon = (ic + np.arange(-n, n + 1)) % lons.size
    lons_out = ((lons[ilon] + 180) % 360) - 180
    if np.any(np.diff(lons_out) < 0):  # Crossing 180 degrees: keep 0..360.
        lons_out = lons[ilon]
    return ilat, ilon, lons_out


def decode(message):
    """
    Decode one GRIB2 message on a regular lat/lon grid.
    Returns (values[lat, lon] with NaN for missing values, latitudes, longitudes).
    """
    import eccodes

    gid = eccodes.codes_new_from_message(message)
    try:
        if eccodes.codes_get(gid, 'gridType') != 'regular_ll':
            raise ValueError('Only regular lat/lon GRIB grids are supported')
        ni, nj = eccodes.codes_get(gid, 'Ni'), eccodes.codes_get(gid, 'Nj')
        lat0 = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
        lat1 = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
        lon0 = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
        dlon = eccodes.codes_get(gid, 'iDirectionIncrementInDegrees')
        values = eccodes.codes_get_values(gid).astype(np.float64)
        if eccodes.codes_get(gid, 'bitmapPresent'):
            values[values == eccodes.codes_get(gid, 'missingValue')] = np.nan
    finally:
        eccodes.codes_release(gid)

    lats = np.linspace(lat0, lat1, nj)
    lons = (lon0 + dlon * np.arange(ni)) % 360
    return values.reshape(nj, ni), lats, lons
