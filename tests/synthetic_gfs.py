"""
Synthetic NOAA GFS archive for offline tests: real GRIB2 messages (written with eccodes) on a
global 1 degree grid, and the matching `.idx` inventories, served by a fake `fetch()`.

Mimics the real `pgrb2.0p25` files: condensate only at 50-1000 hPa, no time averaged
fields in the analysis, fluxes averaged since the start of a 6 hour window, soil fields
missing (bitmap) over sea.
"""

import datetime

import numpy as np

from ls2d.sources.gfs import gfs, pressure_levels

CYCLE = datetime.datetime(2026, 10, 8, 0)
# Regional 1 degree grid (north to south, crossing the 0 degree meridian), to keep the files small.
LATS = np.linspace(64, 40, 25)
LONS = (350.0 + np.arange(31)) % 360
CONDENSATE = ('CLMR', 'ICMR', 'RWMR', 'SNMR', 'GRLE')


def flux_truth(name, hour):
    """
    True mean flux over the interval (hour - 1, hour], W m-2.
    """
    return (100.0 + 10.0 * hour) if name == 'shtfl' else (200.0 + 5.0 * hour)


def _grids():
    lon2d, lat2d = np.meshgrid(LONS, LATS)
    x = ((lon2d - 5.0 + 180) % 360) - 180  # Degrees from 5E.
    y = lat2d - 52.0
    return x, y


def _pl_value(f, p, hour, x, y):
    s = p / 1e5
    if f.grib == 'TMP':
        return 288.0 * s**0.19 + 0.5 * x - 0.4 * y + 0.2 * hour
    if f.grib == 'SPFH':
        return 0.01 * s**3 * (1 + 0.02 * x)
    if f.grib in CONDENSATE:
        return 1e-5 * np.exp(-(((s - 0.85) / 0.05) ** 2)) * np.ones_like(x)
    if f.grib == 'UGRD':
        return 5 + 15 * (1 - s) + 0.5 * y
    if f.grib == 'VGRD':
        return -2 + 3 * (1 - s) - 0.3 * x
    if f.grib == 'VVEL':
        return 0.1 * np.sin(np.pi * s) * (x - y) / 10
    if f.grib == 'HGT':
        return 44330.8 * (1 - (p / 101325) ** 0.1903) + (1 + 0.5 * (1 - s)) * (-15 * y + 8 * x)
    if f.grib == 'O3MR':
        return (1e-5 * np.exp(-(((s - 0.01) / 0.02) ** 2)) + 5e-8) * np.ones_like(x)
    raise KeyError(f.grib)


def _sfc_value(f, hour, x, y):
    land = (x > -2).astype(float)
    if f.name == 'sp':
        return 101000 - 200 * y + 100 * x
    if f.name == 'skt':
        return 290 + 0.3 * x + 0.5 * hour
    if f.name == 't2m':
        return 289 + 0.3 * x + 0.4 * hour
    if f.name == 'q2m':
        return 0.009 * (1 + 0.02 * x)
    if f.name == 'u10':
        return 4 + 0.3 * y
    if f.name == 'v10':
        return -1 - 0.2 * x
    if f.name == 'sfcr':
        return 0.05 + 0.01 * land
    if f.name == 'land':
        return land
    if f.name.startswith('tsoil'):
        return np.where(land > 0.5, 285 + int(f.name[-1]), np.nan)
    if f.name.startswith('soilw'):
        return np.where(land > 0.5, 0.2 + 0.02 * int(f.name[-1]), np.nan)
    raise KeyError(f.name)


def _grib(values):
    import eccodes

    gid = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')
    for key, val in [
        ('Ni', LONS.size), ('Nj', LATS.size),
        ('latitudeOfFirstGridPointInDegrees', LATS[0]), ('latitudeOfLastGridPointInDegrees', LATS[-1]),
        ('longitudeOfFirstGridPointInDegrees', 350.0), ('longitudeOfLastGridPointInDegrees', 20.0),
        ('iDirectionIncrementInDegrees', 1.0), ('jDirectionIncrementInDegrees', 1.0),
        ('packingType', 'grid_ieee'),
    ]:  # fmt: skip
        eccodes.codes_set(gid, key, val)
    v = np.asarray(values, dtype=float).ravel()
    if np.isnan(v).any():
        eccodes.codes_set(gid, 'bitmapPresent', 1)
        eccodes.codes_set(gid, 'missingValue', 9999.0)
        v = np.where(np.isnan(v), 9999.0, v)
    eccodes.codes_set_values(gid, v)
    msg = eccodes.codes_get_message(gid)
    eccodes.codes_release(gid)
    return msg


def make_file(hour, cycle=CYCLE):
    """
    GRIB2 file content and `.idx` text for one forecast hour.
    """
    x, y = _grids()
    fcst = 'anl' if hour == 0 else f'{hour} hour fcst'
    entries = []  # (var, level, fcst, values)

    for f in gfs.registry.fields('pl'):
        for p in pressure_levels:
            if f.grib in CONDENSATE and p < 50:
                continue
            entries.append((f.grib, f'{p:g} mb', fcst, _pl_value(f, p * 100.0, hour, x, y)))

    for f in gfs.registry.fields('sfc'):
        if f.avg:
            if hour == 0:
                continue
            a = 6 * ((hour - 1) // 6)
            mean = np.mean([flux_truth(f.name, h) for h in range(a + 1, hour + 1)])
            entries.append((f.grib, f.level, f'{a}-{hour} hour ave fcst', mean + 0 * x))
        else:
            entries.append((f.grib, f.level, fcst, _sfc_value(f, hour, x, y)))

    blob, idx, offset = b'', [], 0
    for n, (var, level, fc, values) in enumerate(entries, start=1):
        msg = _grib(values)
        idx.append(f'{n}:{offset}:d={cycle:%Y%m%d%H}:{var}:{level}:{fc}:')
        blob += msg
        offset += len(msg)
    return blob, '\n'.join(idx) + '\n'


class FakeArchive:
    """
    In-memory GFS archive (AWS URL layout); `fetch` replaces `ls2d.noaa.gfs_tools.fetch`.
    """

    def __init__(self, hours, cycle=CYCLE):
        import ls2d.noaa.gfs_tools as gt

        self.files = {}
        self.requests = []
        for h in hours:
            url = gt.file_url('AWS', cycle, h)
            blob, idx = make_file(h, cycle)
            self.files[url] = blob
            self.files[f'{url}.idx'] = idx.encode()

    def fetch(self, url, start=None, end=None, **kwargs):
        self.requests.append((url, start, end))
        if url not in self.files:
            raise FileNotFoundError(url)
        data = self.files[url]
        if start is None:
            return data
        return data[start : None if end is None else end + 1]


def settings(path, start_hour=0, end_hour=3):
    return dict(
        central_lat=51.97,
        central_lon=4.93,
        area_size=3.0,
        case_name='synthetic',
        gfs_path=str(path),
        start_date=CYCLE + datetime.timedelta(hours=start_hour),
        end_date=CYCLE + datetime.timedelta(hours=end_hour),
        source='gfs',
    )
