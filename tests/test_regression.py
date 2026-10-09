"""
New pipeline vs. golden output of the pre-refactor code (see `make_golden.py`).
"""

import os

import numpy as np
import pytest
import xarray as xr

import ls2d
from conftest import DATA

z = np.arange(10, 5000, 50).astype(float)

# Physics deliberately taken from the ARCO path of `develop_arco` instead of the legacy `Read_era5`:
#   h2o_lay : water vapour VMR from qv instead of qt.
#   wq      : divided by the surface density (legacy returned the mass flux kg m-2 s-1).
ARCO_PHYSICS = {'h2o_lay', 'wq'}


def assert_var_close(new, ref, v, rtol, atol_rel=0):
    """
    `atol_rel`: absolute tolerance relative to the maximum of the field, for float32 input
    where near-zero values (e.g. `wls`) have large relative round-off errors.
    """
    assert new[v].dims == ref[v].dims, v
    atol = atol_rel * float(np.abs(ref[v]).max())
    np.testing.assert_allclose(new[v].values, ref[v].values, rtol=rtol, atol=atol, err_msg=v)


def golden(name):
    return xr.open_dataset(os.path.join(DATA, f'golden_{name}.nc'))


def assert_close(new, ref, rtol, skip=(), long_names=True, atol_rel=0):
    """
    `long_names=False` for the `develop_arco` golden file, which renamed some long names;
    the long names of the released version (legacy golden files) are kept.
    """
    assert set(new.data_vars) == set(ref.data_vars)
    for v in ref.data_vars:
        if v in skip:
            continue
        assert_var_close(new, ref, v, rtol, atol_rel)
        assert new[v].attrs['units'] == ref[v].attrs['units'], v
        if long_names:
            assert new[v].attrs['long_name'] == ref[v].attrs['long_name'], v
    for c in ('z', 'zs', 'lay', 'lev'):
        np.testing.assert_array_equal(new[c].values, ref[c].values)
    np.testing.assert_array_equal(new.time.values, ref.time.values)


@pytest.mark.parametrize('method', ['2nd', '4th'])
def test_read_era5_wrapper_vs_legacy(settings, method):
    era = ls2d.Read_era5(settings)
    era.calculate_forcings(n_av=1, method=method)
    les = era.get_les_input(z)

    assert_close(les, golden(f'legacy_{method}'), rtol=1e-10, skip=ARCO_PHYSICS)
    # Deliberately changed variables agree with the ARCO path (float32 input there).
    for v in ARCO_PHYSICS:
        assert_var_close(les, golden('arco'), v, rtol=1e-5)


@pytest.mark.parametrize('method', ['2nd', '4th'])
def test_stateless_pipeline_vs_legacy(settings, method):
    raw = ls2d.read_era5(settings)
    column = ls2d.calculate_forcings(raw, n_av=1, method=method)
    les = ls2d.get_les_input(column, z)
    assert_close(les, golden(f'legacy_{method}'), rtol=1e-10, skip=ARCO_PHYSICS)


def test_arco_vs_golden(settings):
    # ARCO files are float32: compare at single precision.
    raw = ls2d.read_era5(dict(settings, data_source='ARCO'))
    les = ls2d.get_les_input(ls2d.calculate_forcings(raw, n_av=1), z)
    assert_close(les, golden('arco'), rtol=1e-5, long_names=False, atol_rel=1e-6)


def test_create_column_input_compat(settings):
    # `develop_arco` API: field quantities on 3D grid -> LES input.
    ds_3d = ls2d.read_era5_arco(settings)
    les = ls2d.create_column_input(ds_3d, z, n_av=1)
    assert_close(les, golden('arco'), rtol=1e-5, long_names=False, atol_rel=1e-6)


def test_global_attributes(settings):
    raw = ls2d.read_era5(settings)
    les = ls2d.get_les_input(ls2d.calculate_forcings(raw, n_av=1), z)
    ref = golden('arco')
    for attr in ('fc', 'central_lat', 'central_lon', 'area', 'description', 'reference'):
        assert les.attrs[attr] == ref.attrs[attr], attr
    assert les.attrs['source'] == 'ERA5 (CDS) + (LS)²D'


def test_small_domain_vs_legacy(era5_path, tmp_path):
    """
    3x3 grid points (e.g. `area_size=0.25`) with `n_av=1`: one-sided gradients at the edges, as in the legacy code.
    """
    import synthetic_era5 as syn

    syn.write_cds(tmp_path, n=1, case='small')
    settings = syn.settings(tmp_path, case='small')

    era = ls2d.Read_era5(settings)
    era.calculate_forcings(n_av=1, method='2nd')
    assert_close(era.get_les_input(z), golden('legacy_small'), rtol=1e-10, skip=ARCO_PHYSICS)

    with pytest.raises(ValueError, match='method="2nd"'):
        era.calculate_forcings(n_av=1, method='4th')
    with pytest.raises(ValueError, match='Domain too small for n_av=2'):
        era.calculate_forcings(n_av=2, method='2nd')
