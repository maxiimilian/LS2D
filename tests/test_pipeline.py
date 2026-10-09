import copy
import importlib

import numpy as np
import pytest
import xarray as xr

import ls2d
from ls2d.column import validate

# `ls2d.ecmwf.download_era5` is shadowed by the function of the same name.
dl = importlib.import_module('ls2d.ecmwf.download_era5')

z = np.arange(10, 3000, 100).astype(float)


def test_settings_not_modified(settings):
    before = copy.deepcopy(settings)
    era = ls2d.Read_era5(settings)
    era.calculate_forcings(n_av=1)
    era.get_les_input(z)
    assert settings == before


def test_raw_dataset_conventions(settings):
    raw = ls2d.read_era5(settings)
    validate(raw)
    assert raw['ml:t'].dims == ('time', 'level', 'latitude', 'longitude')
    assert raw['pl:z'].dims == ('time', 'pressure_level', 'latitude', 'longitude')
    assert raw.pressure_level[0] == 100000.0
    assert raw.time.size == 13  # 06 to 18 UTC


def test_calculate_forcings_is_pure(settings):
    raw = ls2d.read_era5(settings)
    before = raw.copy(deep=True)
    c1 = ls2d.calculate_forcings(raw, n_av=1, method='4th')
    c2 = ls2d.calculate_forcings(raw, n_av=1, method='4th')
    assert raw.identical(before)
    xr.testing.assert_identical(c1, c2)


def test_les_input_needs_column_variables(settings):
    raw = ls2d.read_era5(settings, outputs=['ps'])
    column = ls2d.calculate_forcings(raw, outputs=['ps'])
    assert list(column.data_vars) == ['ps']
    les = ls2d.get_les_input(column, z)
    assert list(les.data_vars) == ['ps', 'time_sec']
    with pytest.raises(KeyError, match='calculate_forcings'):
        ls2d.get_les_input(column, z, outputs=['thl'])


def test_wrapper_requires_calculate_forcings(settings):
    era = ls2d.Read_era5(settings)
    with pytest.raises(RuntimeError, match='calculate_forcings'):
        era.get_les_input(z)


def test_subset_reads_only_needed_fields(settings):
    raw = ls2d.read_era5(settings, outputs=['ps', 'ts'])
    assert set(raw.data_vars) == {'sfc:sp', 'sfc:skt'}
    col = ls2d.calculate_forcings(raw, n_av=1, outputs=['ps', 'ts'])
    assert set(col.data_vars) == {'ps', 'ts'}

    # As LES output, `ug` and `vg` are interpolated to the LES grid, which needs the
    # model level heights `z`, and therefore the full thermodynamics on model levels.
    keys = {f.key for f in ls2d.required_era5_fields(['ug', 'vg'])}
    assert {'pl:z', 'sfc:sp', 'ml:t', 'ml:q'} <= keys and 'ml:u' not in keys
    outputs = ['ug']
    column = ls2d.calculate_forcings(ls2d.read_era5(settings, outputs=outputs), outputs=outputs)
    assert set(ls2d.get_les_input(column, z, outputs=outputs).data_vars) == {'ug', 'time_sec'}
    # Without `outputs`, everything the column dataset supports is returned.
    assert set(ls2d.get_les_input(column, z).data_vars) == {'ug', 'z_lay', 'time_sec'}


def test_column_only_quantities(settings):
    raw = ls2d.read_era5(settings)
    col = ls2d.calculate_forcings(raw, n_av=1, outputs=['dtu_total', 'dtu_advec', 'dtu_coriolis'])
    np.testing.assert_allclose(col.dtu_total, col.dtu_advec + col.dtu_coriolis)


def test_domain_too_small(settings):
    raw = ls2d.read_era5(settings)
    with pytest.raises(ValueError, match='Domain too small'):
        ls2d.calculate_forcings(raw, n_av=5, method='4th')


def test_add_new_era5_variable(settings, era5_path, clean_registry, monkeypatch, tmp_path):
    """
    The use case of the registry: add a new ERA5 field and a forcing derived from it.
    """
    import shutil

    # Work on a copy of the synthetic data, as we modify the surface file.
    shutil.copytree(era5_path, tmp_path / 'era5')
    settings = dict(settings, era5_path=str(tmp_path / 'era5'))

    ls2d.era5.field('t2m', 'sfc', '167.128', cds='2m_temperature', arco='2m_temperature', units='K')

    @ls2d.era5.quantity('t2m', requires=('sfc:t2m',), units='K', long_name='2 m temperature', reduce='mean')
    def t2m(t2m, ctx):
        return t2m

    ls2d.les_output('t2m', 't2m', '2 m temperature', 'K')

    assert [f.key for f in ls2d.required_era5_fields(['t2m'])] == ['sfc:t2m']

    # 1. Existing files miss the new field: download re-requests the surface file only,
    #    including the fields already in it.
    queue = []
    monkeypatch.setattr(dl, 'cdsapi', object())
    monkeypatch.setattr(dl, '_download_era5_file', lambda req: queue.append(req) or True)
    ls2d.download_era5(settings)
    assert [q['ftype'] for q in queue] == ['surface_an']
    keys = {f.key for f in queue[0]['fields']}
    assert 'sfc:t2m' in keys and 'sfc:sp' in keys and len(keys) == 23

    # Reading without the new field present gives a clear error.
    with pytest.raises(KeyError, match='sfc:t2m'):
        ls2d.read_era5(settings, outputs=['t2m'])

    # 2. "Download" by adding the field to the surface file.
    f = tmp_path / 'era5' / 'synthetic' / '2016' / '08' / '15' / 'surface_an.nc'
    ds = xr.open_dataset(f).load()
    ds.close()
    ds['t2m'] = ds['skt'] - 1.0
    ds.to_netcdf(f, format='NETCDF4_CLASSIC')

    queue.clear()
    ls2d.download_era5(settings)
    assert queue == []

    # 3. Process.
    raw = ls2d.read_era5(settings, outputs=['t2m'])
    les = ls2d.get_les_input(ls2d.calculate_forcings(raw, n_av=1, outputs=['t2m']), z)
    ts = ls2d.get_les_input(
        ls2d.calculate_forcings(ls2d.read_era5(settings, outputs=['ts']), n_av=1, outputs=['ts']), z
    )
    np.testing.assert_allclose(les.t2m, ts.ts - 1.0)
    assert les.t2m.attrs['long_name'] == '2 m temperature'


def test_download_skips_complete_files(settings, monkeypatch):
    queue = []
    monkeypatch.setattr(dl, 'cdsapi', object())
    monkeypatch.setattr(dl, '_download_era5_file', lambda req: queue.append(req) or True)
    assert ls2d.download_era5(settings)
    assert queue == []


def test_arco_field_checks(clean_registry):
    arco = importlib.import_module('ls2d.google.download_era5_arco')
    fields = ls2d.required_era5_fields()
    assert arco._arco_fields(fields) == fields

    ls2d.era5.field('no_arco', 'sfc', '1.128', cds='no_arco')
    with pytest.raises(ValueError, match='not available in Google ARCO'):
        arco._arco_fields([clean_registry['sfc:no_arco']])

    ls2d.era5.field('z', 'sfc', '129.128', cds='geopotential', arco='geopotential_at_surface')
    with pytest.raises(ValueError, match='same short name'):
        arco._arco_fields([clean_registry['sfc:z'], clean_registry['pl:z']])
