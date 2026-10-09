"""
A second, synthetic data source ("plev_test") to check that LS2D is source-agnostic.

It mimics a GFS-like product: all atmospheric fields on pressure levels only (no hybrid
model levels), one condensate species, surface fluxes positive upward in W m-2, and no
HTESSEL land surface fields. The data is the same synthetic atmosphere as `synthetic_era5`,
interpolated from the ERA5 model levels to pressure levels, so both sources should give
(nearly) the same LES input.
"""

import os

import numpy as np
import xarray as xr

import ls2d.forcing.constants as c
from ls2d.forcing import Source, register_source, get_source
from ls2d.forcing.raw import standardize
from ls2d.forcing.vertical import to_terrain_following

import synthetic_era5 as syn

NAME = 'plev_test'
FILE = 'plev.nc'

# Terrain-following target levels (fraction of surface pressure), surface to top.
SIGMA = np.sort(syn.PRESSURE_LEVELS / 1000.0)[::-1]


def write_plev(path, date=syn.DATE):
    """
    Write `{path}/{case}/yyyy/mm/dd/plev.nc`, with GFS-like names and conventions.
    """
    times, lat, lon, f = syn._fields(date)
    ifs = syn.ifs

    # Full level pressure of the synthetic ERA5 model levels, top->bottom like the fields.
    ph = ifs.a[None, :, None, None] + ifs.b[None, :, None, None] * f['sp'][:, None]
    ph[:, -1] = 0.34
    p_ml = (0.5 * (ph[:, 1:] + ph[:, :-1]))[:, ::-1]  # top->bottom

    p_pl = syn.PRESSURE_LEVELS[::-1] * 100.0  # 1000 hPa first, as in GFS files

    def to_pl(a):
        out = np.empty((a.shape[0], p_pl.size) + a.shape[2:])
        for t in range(a.shape[0]):
            for j in range(a.shape[2]):
                for i in range(a.shape[3]):
                    out[t, :, j, i] = np.interp(np.log(p_pl), np.log(p_ml[t, :, j, i]), a[t, :, j, i])
        return out

    d3 = ('time', 'isobaricInhPa', 'latitude', 'longitude')
    d2 = ('time', 'latitude', 'longitude')
    ql = f['clwc'] + f['ciwc'] + f['crwc'] + f['cswc']

    ds = xr.Dataset(
        {
            't': (d3, to_pl(f['t'])),
            'q': (d3, to_pl(f['q'])),
            'clwmr': (d3, to_pl(ql)),
            'u': (d3, to_pl(f['u'])),
            'v': (d3, to_pl(f['v'])),
            'w': (d3, to_pl(f['w'])),
            'gh': (d3, f['z_pl'][:, ::-1] / c.grav),
            'sp': (d2, f['sp']),
            'skt': (d2, f['skt']),
            'shtfl': (d2, -f['ishf']),
            'lhtfl': (d2, -f['ie'] * c.Lv),
        },
        coords=dict(
            time=np.array(times, dtype='datetime64[ns]'),
            isobaricInhPa=('isobaricInhPa', p_pl / 100.0),
            latitude=('latitude', lat),
            longitude=('longitude', lon),
        ),
    )
    d = syn._day_dir(path, date)
    ds.to_netcdf(os.path.join(d, FILE))


def read_plev(settings, outputs=None, fields=None):
    src = get_source(NAME)
    fields = src.required_fields(outputs) if fields is None else fields
    d = syn._day_dir(settings['era5_path'], syn.DATE, settings['case_name'])
    ds = xr.open_dataset(os.path.join(d, FILE)).sel(time=slice(settings['start_date'], settings['end_date']))
    attrs = dict(
        central_lat=settings['central_lat'], central_lon=settings['central_lon'], source='synthetic', ls2d_source=NAME
    )
    return standardize({'pl': ds, 'sfc': ds}, fields, attrs, pressure_level_dim='isobaricInhPa').load()


def make_source():
    """
    Create and register the source. Everything source specific is in here; the
    rest (thl, z, ph, advection, LES input, ...) comes from the LS2D core.
    """
    src = Source(NAME, 'synthetic pressure level data', read=read_plev)

    # fmt: off
    for name, units in [('t', 'K'), ('q', 'kg kg-1'), ('clwmr', 'kg kg-1'), ('u', 'm s-1'), ('v', 'm s-1'),
                        ('w', 'Pa s-1'), ('gh', 'gpm')]:
        src.field(name, 'pl', units=units)
    for name, units in [('sp', 'Pa'), ('skt', 'K'), ('shtfl', 'W m-2'), ('lhtfl', 'W m-2')]:
        src.field(name, 'sfc', units=units)
    # fmt: on

    def on_levels(name):
        @src.quantity(name[0], requires=(f'pl:{name[1]}', 'ps'))
        def f(da, ps, ctx):
            return to_terrain_following(da, ps, SIGMA)

    for name in [('T', 't'), ('qv', 'q'), ('ql', 'clwmr'), ('u', 'u'), ('v', 'v'), ('omega', 'w')]:
        on_levels(name)

    @src.quantity('p', requires=('ps',))
    def p(ps, ctx):
        sigma = xr.DataArray(SIGMA, dims='level')
        return (sigma * ps).transpose('time', 'level', ...)

    # `ph` (half level pressure) is not provided: the core derives it from `p` and `ps`.

    @src.quantity('phi_p', requires=('pl:gh',))
    def phi_p(gh, ctx):
        return gh * c.grav

    @src.quantity('ps', requires=('sfc:sp',))
    def ps(sp, ctx):
        return sp

    @src.quantity('ts', requires=('sfc:skt',))
    def ts(skt, ctx):
        return skt

    @src.quantity('wth', requires=('sfc:shtfl', 'ps', 'rhos'))
    def wth(shtfl, ps, rhos, ctx):
        return shtfl / (rhos * c.cpd * (ps / c.p0) ** (c.Rd / c.cpd))

    @src.quantity('wq', requires=('sfc:lhtfl', 'rhos'))
    def wq(lhtfl, rhos, ctx):
        return lhtfl / (rhos * c.Lv)

    return register_source(src, replace=True)
