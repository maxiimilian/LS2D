"""
Deterministic synthetic ERA5 files, written in the on-disk layouts that LS2D reads:
    - CDS/MARS: `{path}/{case}/yyyy/mm/dd/{model_an,pressure_an,surface_an}.nc`
    - Google ARCO: `{path}/{case}/yyyy/mm/dd/era5_arco.nc`

The fields are smooth and physically plausible (L137 reference atmosphere with
horizontal gradients and a diurnal cycle), so that advective tendencies, geostrophic
wind etc. are non-trivial. No downloads or credentials needed.
"""

import datetime
import os

import numpy as np
import xarray as xr

from ls2d.ecmwf.IFS_tools import IFS_tools

ifs = IFS_tools('L137')

CENTRAL_LAT = 51.97
CENTRAL_LON = 4.93
CASE = 'synthetic'
DATE = datetime.datetime(2016, 8, 15)

# 0.25 deg grid around nearest point (52.0, 5.0); 13x13 points.
_n = 6
LATS = 52.0 + 0.25 * np.arange(-_n, _n + 1)
LONS = 5.0 + 0.25 * np.arange(-_n, _n + 1)
PRESSURE_LEVELS = np.array(
    [
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
        50,
        70,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    ]
)


def _subset(lat, lon, f, n):
    """
    Keep only +/- `n` grid points around the central point (for small download areas).
    """
    if n is None:
        return lat, lon, f
    jc, ic = lat.size // 2, lon.size // 2
    sj, si = slice(jc - n, jc + n + 1), slice(ic - n, ic + n + 1)
    return lat[sj], lon[si], {k: v[..., sj, si] for k, v in f.items()}


def _fields(date):
    """
    All raw ERA5 fields with dims (time, [level], lat, lon); `level` top->bottom,
    latitude north->south (as in the ERA5 files).
    """
    nt = 24
    hours = np.arange(nt)
    lat = LATS[::-1]
    lon = LONS

    # Normalised coordinates.
    t = hours[:, None, None, None] / 24.0
    y = lat[None, None, :, None] - 52.0
    x = lon[None, None, None, :] - 5.0
    diurnal = np.sin(2 * np.pi * (t - 0.25))

    # Reference profiles, top->bottom (IFS_tools stores them bottom->top).
    T_ref = ifs.T[::-1][None, :, None, None]
    pf_ref = ifs.pf[::-1][None, :, None, None]
    sig = pf_ref / 101325.0  # ~0 at top, ~1 at surface

    f = {}
    f['t'] = T_ref + sig * (2.0 * x - 1.5 * y + 3.0 * diurnal * sig**4)
    f['q'] = 0.012 * sig**3 * (1 + 0.05 * x - 0.08 * y + 0.1 * diurnal)
    f['clwc'] = 1e-5 * np.exp(-(((sig - 0.85) / 0.05) ** 2)) * (1 + 0.2 * x)
    f['ciwc'] = 5e-6 * np.exp(-(((sig - 0.4) / 0.05) ** 2)) * (1 - 0.2 * y)
    f['crwc'] = 1e-6 * np.exp(-(((sig - 0.95) / 0.03) ** 2)) * (1 + 0.1 * t)
    f['cswc'] = 2e-6 * np.exp(-(((sig - 0.6) / 0.05) ** 2)) * (1 + 0.1 * x * y)
    f['u'] = 5 + 20 * (1 - sig) + 1.5 * y + 0.5 * x + diurnal * sig
    f['v'] = -2 + 5 * (1 - sig) - 1.0 * x + 0.3 * y**2 - 0.5 * diurnal
    f['w'] = 0.05 * np.sin(np.pi * sig) * (x - y + 0.5 * diurnal)
    f['o3'] = 1e-5 * np.exp(-(((sig - 0.01) / 0.02) ** 2)) + 5e-8 * (1 + 0.01 * x)
    f = {k: np.broadcast_to(v, (nt, 137, lat.size, lon.size)).astype(np.float64) for k, v in f.items()}

    # Pressure level geopotential: standard atmosphere tilted horizontally (-> geostrophic wind).
    p_pl = PRESSURE_LEVELS[None, :, None, None] * 100.0
    zref = np.interp(np.log(PRESSURE_LEVELS * 100.0), np.log(ifs.pf), ifs.gpa)[None, :, None, None]
    tilt = 1 + zref / 8000.0
    z = (zref + tilt * (-15.0 * y + 8.0 * x + 2.0 * x * y + 3.0 * diurnal)) * ifs.grav
    f['z_pl'] = np.broadcast_to(z, (nt, PRESSURE_LEVELS.size, lat.size, lon.size)).astype(np.float64)
    del p_pl

    # Surface fields.
    t2 = t[:, 0]
    y2 = y[:, 0]
    x2 = x[:, 0]
    d2 = diurnal[:, 0]
    shp = (nt, lat.size, lon.size)

    def b(a):
        return np.broadcast_to(a, shp).astype(np.float64)

    land = (x2 > -1.0) | (y2 < 0.5)
    land = np.broadcast_to(land, shp)

    f['sp'] = b(101000 - 300 * y2 + 150 * x2 - 80 * d2)
    f['skt'] = b(290 + 6 * d2 + 0.5 * x2)
    f['sst'] = b(289 + 0.2 * x2 - 0.3 * y2)
    f['ishf'] = b(-150 * np.maximum(d2, 0) - 10 * x2)
    f['ie'] = b(-8e-5 * np.maximum(d2, 0) * (1 + 0.1 * y2))
    f['fsr'] = b(0.1 + 0.02 * x2**2)
    f['flsr'] = b(np.log(0.01 + 0.002 * y2**2))
    f['slt'] = np.where(land, 3.0, 0.0) + 0 * t2
    f['tvl'] = np.where(land, 2.0, 0.0) + 0 * t2
    f['tvh'] = np.where(land, 5.0, 0.0) + 0 * t2
    f['lai_lv'] = b(2.5 + 0.1 * x2)
    f['lai_hv'] = b(4.0 - 0.1 * y2)
    f['cvl'] = b(0.6 + 0.05 * x2)
    f['cvh'] = b(0.3 - 0.05 * y2)
    for i in range(1, 5):
        f[f'stl{i}'] = b(288 + 2 * d2 / i + 0.1 * i * x2)
        f[f'swvl{i}'] = b(0.3 + 0.02 * i + 0.01 * y2)

    times = [date + datetime.timedelta(hours=int(h)) for h in hours]
    return times, lat, lon, f


ML_VARS = ['u', 'v', 'w', 't', 'q', 'clwc', 'ciwc', 'crwc', 'cswc', 'o3']
SFC_VARS = [
    'sst',
    'skt',
    'ishf',
    'ie',
    'fsr',
    'flsr',
    'sp',
    'slt',
    'tvl',
    'tvh',
    'lai_lv',
    'lai_hv',
    'cvl',
    'cvh',
    'stl1',
    'stl2',
    'stl3',
    'stl4',
    'swvl1',
    'swvl2',
    'swvl3',
    'swvl4',
]


def _day_dir(path, date, case=CASE):
    d = os.path.join(path, case, f'{date.year:04d}', f'{date.month:02d}', f'{date.day:02d}')
    os.makedirs(d, exist_ok=True)
    return d


def write_cds(path, date=DATE, n=None, case=CASE):
    """
    Write `model_an.nc`, `pressure_an.nc`, `surface_an.nc` in the (patched) CDS layout.
    `n` limits the domain to +/- n grid points around the centre.
    """
    times, lat, lon, f = _fields(date)
    lat, lon, f = _subset(lat, lon, f, n)
    hours = np.array([(t - datetime.datetime(1900, 1, 1)).total_seconds() / 3600.0 for t in times])
    d = _day_dir(path, date, case)

    def coords(**extra):
        c = dict(
            time=('time', hours, {'units': 'hours since 1900-01-01 00:00:00.0'}),
            latitude=('latitude', lat),
            longitude=('longitude', lon),
        )
        c.update(extra)
        return c

    d3 = ('time', 'level', 'latitude', 'longitude')
    d2 = ('time', 'latitude', 'longitude')

    ml = xr.Dataset({v: (d3, f[v]) for v in ML_VARS}, coords=coords(level=('level', np.arange(1, 138, dtype=np.int32))))
    pl = xr.Dataset({'z': (d3, f['z_pl'])}, coords=coords(level=('level', PRESSURE_LEVELS.astype(np.int32))))
    sfc = xr.Dataset({v: (d2, f[v]) for v in SFC_VARS}, coords=coords())

    for name, ds in [('model_an', ml), ('pressure_an', pl), ('surface_an', sfc)]:
        ds.to_netcdf(os.path.join(d, f'{name}.nc'), format='NETCDF4_CLASSIC')


def write_arco(path, date=DATE):
    """
    Write `era5_arco.nc` in the layout of `download_era5_arco()`.
    """
    times, lat, lon, f = _fields(date)
    d = _day_dir(path, date)

    d3 = ('time', 'level', 'latitude', 'longitude')
    d3p = ('time', 'pressure_level', 'latitude', 'longitude')
    d2 = ('time', 'latitude', 'longitude')

    variables = {v: (d3, f[v].astype(np.float32)) for v in ML_VARS}
    variables['z'] = (d3p, f['z_pl'].astype(np.float32))
    variables.update({v: (d2, f[v].astype(np.float32)) for v in SFC_VARS})

    ds = xr.Dataset(
        variables,
        coords=dict(
            time=np.array(times, dtype='datetime64[ns]'),
            level=('level', np.arange(1, 138, dtype=np.int32)),
            pressure_level=('pressure_level', PRESSURE_LEVELS.astype(np.int32)),
            latitude=('latitude', lat),
            longitude=('longitude', lon),
        ),
    )
    ds.to_netcdf(os.path.join(d, 'era5_arco.nc'))


def settings(path, case=CASE):
    return {
        'central_lat': CENTRAL_LAT,
        'central_lon': CENTRAL_LON,
        'area_size': 1.0,
        'case_name': case,
        'era5_path': str(path),
        'start_date': DATE + datetime.timedelta(hours=6),
        'end_date': DATE + datetime.timedelta(hours=18),
        'data_source': 'CDS',
        'write_log': False,
    }
