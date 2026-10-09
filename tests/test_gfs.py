"""
GFS source, offline: synthetic GRIB2 archive (`synthetic_gfs`) served through a fake `fetch()`.
A live test against the real NOAA archive runs with `LS2D_LIVE=1`.
"""

import datetime
import os

import numpy as np
import pytest
import xarray as xr

import ls2d
import ls2d.noaa.gfs_tools as gt
import synthetic_gfs as sg
from ls2d.noaa.download_gfs import gfs_file
from ls2d.noaa.read_gfs import interval_means

z = np.arange(10, 5000, 50).astype(float)


@pytest.fixture(scope='module')
def archive():
    return sg.FakeArchive(hours=range(0, 8))


@pytest.fixture
def fake(archive, monkeypatch):
    archive.requests.clear()
    monkeypatch.setattr(gt, 'fetch', archive.fetch)
    return archive


#
# Helpers.
#
def test_cycle_and_hours():
    s = dict(start_date=datetime.datetime(2026, 10, 8, 7), end_date=datetime.datetime(2026, 10, 8, 10))
    assert gt.cycle(s) == datetime.datetime(2026, 10, 8, 6)
    assert gt.forecast_hours(s) == [1, 2, 3, 4]

    s['gfs_cycle'] = datetime.datetime(2026, 10, 8, 0)
    assert gt.forecast_hours(s) == [7, 8, 9, 10]

    # 3-hourly after +120 h.
    s = dict(gfs_cycle=datetime.datetime(2026, 10, 1), start_date=datetime.datetime(2026, 10, 5, 23),
             end_date=datetime.datetime(2026, 10, 6, 6))  # fmt: skip
    assert gt.forecast_hours(s) == [119, 120, 123, 126]
    assert gt.previous_hour(123, s) == 120 and gt.previous_hour(119, s) == 118

    with pytest.raises(ValueError, match='GFS output times'):
        gt.forecast_hours(dict(s, start_date=datetime.datetime(2026, 10, 6, 1)))
    with pytest.raises(ValueError, match='06, 12, or 18'):
        gt.cycle(dict(s, gfs_cycle=datetime.datetime(2026, 10, 1, 3)))


def test_idx_and_ranges():
    msgs = gt.parse_idx('1:0:d=2026100800:TMP:850 mb:6 hour fcst:\n2:100:d=2026100800:TMP:900 mb:6 hour fcst:\n'
                        '3:250:d=2026100800:SHTFL:surface:0-6 hour ave fcst:\n4:400:d=2026100800:LAND:surface:6 hour fcst:\n')  # fmt: skip
    assert [(m['start'], m['end']) for m in msgs] == [(0, 99), (100, 249), (250, 399), (400, None)]
    assert gt.avg_start(msgs[2]['fcst']) == 0 and gt.avg_start(msgs[0]['fcst']) is None
    # Adjacent messages are merged into one request.
    assert [(s, e, len(g)) for s, e, g in gt.merge_ranges([msgs[0], msgs[1], msgs[3]])] == [(0, 249, 2), (400, None, 1)]


def test_interval_means():
    # Averages since the start of 6 h windows -> hourly means.
    true = {h: 100.0 + 10 * h for h in range(1, 13)}
    hours = list(range(3, 13))
    starts = [6 * ((h - 1) // 6) for h in hours]
    avgs = [np.mean([true[k] for k in range(a + 1, h + 1)]) for h, a in zip(hours, starts)]
    out = interval_means(np.array(avgs), hours, starts)
    assert np.isnan(out[0])  # Needs f002.
    np.testing.assert_allclose(out[1:], [true[h] for h in hours[1:]])

    # 3-hourly windows after +120 h: the average is the interval mean.
    out = interval_means(np.array([1.0, 2.0, 3.0]), [120, 123, 126], [114, 120, 123])
    np.testing.assert_allclose(out[1:], [2.0, 3.0])

    # Analysis: not available.
    assert np.isnan(interval_means(np.array([np.nan, 5.0]), [0, 1], [-1, 0])[0])


def test_crop_across_meridian():
    lats = np.linspace(90, -90, 721)
    lons = np.arange(1440) * 0.25
    ilat, ilon, lon_out = gt.crop_indices(lats, lons, 51.97, 0.1, 0.5)
    np.testing.assert_allclose(lon_out, [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(lats[ilat], [52.75, 52.5, 52.25, 52.0, 51.75, 51.5, 51.25])


#
# Download -> read -> LES, against the synthetic archive.
#
def test_download_read_les(fake, tmp_path, caplog):
    settings = sg.settings(tmp_path, 0, 3)
    ls2d.download(settings)

    for h in range(4):
        assert os.path.isfile(gfs_file(settings, sg.CYCLE, h))
    n_requests = len(fake.requests)

    # Second call: nothing to download.
    ls2d.download(settings)
    assert len(fake.requests) == n_requests

    raw = ls2d.read(settings)
    assert raw.attrs['ls2d_source'] == 'gfs'
    assert raw['pl:t'].dims == ('time', 'pressure_level', 'latitude', 'longitude')
    assert raw.pressure_level[0] == 100000.0 and np.all(np.diff(raw.latitude) > 0)
    assert raw.time.size == 4

    # Condensate is zero above 50 hPa (not in the GRIB files).
    assert float(raw['pl:clwmr'].sel(pressure_level=1000.0).max()) == 0.0

    # Fluxes: hourly interval means; analysis filled with the first interval.
    for name in ('shtfl', 'lhtfl'):
        flux = raw[f'sfc:{name}'].isel(latitude=0, longitude=0).values
        np.testing.assert_allclose(flux, [sg.flux_truth(name, h) for h in (1, 1, 2, 3)])
    assert 'using the interval mean of f001' in caplog.text

    column = ls2d.calculate_forcings(raw, n_av=1, method='4th')
    les = ls2d.get_les_input(column, z)
    assert les.attrs['ls2d_source'] == 'gfs'
    assert set(les.data_vars) == set(ls2d.gfs.default_outputs()) | {'time_sec'}
    for v in les.data_vars:
        if v != 'sst':
            assert np.isfinite(les[v]).all(), v
    # The averaging area is land only: SST undefined, as in ERA5.
    assert np.isnan(les.sst).all()

    # Surface anchor: lowest level = 2 m values; sigma = 1 at the surface.
    np.testing.assert_allclose(column.p.isel(level=0), column.ps)
    t2m = raw['sfc:t2m'].isel(latitude=slice(3, 6), longitude=slice(3, 6)).mean(('latitude', 'longitude'))
    np.testing.assert_allclose(column['T'].isel(level=0), t2m)
    # Soil: averaged over land points only (NaN over sea).
    assert np.isfinite(column.t_soil).all() and column.t_soil.dims == ('time', 'soil_layer')


def test_les_input_same_definition_as_era5(fake, tmp_path, settings):
    gfs = ls2d.get_les_input(ls2d.calculate_forcings(ls2d.read(_downloaded(tmp_path)), n_av=1), z)
    era5 = ls2d.get_les_input(ls2d.calculate_forcings(ls2d.read_era5(settings), n_av=1), z)
    for v in gfs.data_vars:
        assert gfs[v].dims == era5[v].dims, v
        assert gfs[v].attrs == era5[v].attrs, v


def _downloaded(path, start=0, end=3):
    s = sg.settings(path, start, end)
    ls2d.download(s)
    return s


def test_previous_hour_for_fluxes(fake, tmp_path):
    # Starting at f003: f002 is needed (fluxes only) to compute the 2-3 h mean.
    settings = _downloaded(tmp_path, 3, 5)
    prev = xr.open_dataset(gfs_file(settings, sg.CYCLE, 2))
    assert set(prev.data_vars) == {'shtfl', 'lhtfl', 'shtfl_avg_start', 'lhtfl_avg_start'}

    raw = ls2d.read(settings)
    flux = raw['sfc:shtfl'].isel(latitude=0, longitude=0).values
    np.testing.assert_allclose(flux, [sg.flux_truth('shtfl', h) for h in (3, 4, 5)])


def test_only_requested_fields(fake, tmp_path):
    settings = sg.settings(tmp_path, 1, 2)
    ls2d.download(settings, outputs=['ps', 'ts'])
    ds = xr.open_dataset(gfs_file(settings, sg.CYCLE, 1))
    assert set(ds.data_vars) == {'sp', 'skt'}

    # Asking for more: files are downloaded again, keeping the fields already there.
    ls2d.download(settings, outputs=['ps', 'ts', 'u'])
    ds = xr.open_dataset(gfs_file(settings, sg.CYCLE, 1))
    assert {'sp', 'skt', 'u', 'u10'} <= set(ds.data_vars)


def test_missing_cycle(fake, tmp_path):
    settings = sg.settings(tmp_path, 0, 3)
    settings['gfs_cycle'] = datetime.datetime(2026, 10, 7, 18)
    settings['start_date'] = datetime.datetime(2026, 10, 7, 18)
    with pytest.raises(FileNotFoundError, match='not available on AWS'):
        ls2d.download(settings)


#
# Live test against the real NOAA archive (network, ~1 GB): LS2D_LIVE=1 pytest tests/test_gfs.py
#
@pytest.mark.skipif(os.environ.get('LS2D_LIVE') != '1', reason='set LS2D_LIVE=1 to download real GFS data')
def test_live_gfs(tmp_path):
    cycle = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(days=2)
    cycle = cycle.replace(hour=0, minute=0, second=0, microsecond=0)
    settings = dict(central_lat=51.97, central_lon=4.93, area_size=0.75, case_name='live', gfs_path=str(tmp_path),
                    start_date=cycle, end_date=cycle + datetime.timedelta(hours=2), source='gfs')  # fmt: skip
    ls2d.download(settings)
    les = ls2d.get_les_input(ls2d.calculate_forcings(ls2d.read(settings), n_av=1), z)
    assert 270 < float(les.thl.isel(z=0).mean()) < 310
    for v in les.data_vars:
        assert np.isfinite(les[v]).all(), v
