"""
Source abstraction: a second (pressure level only, GFS-like) source must give
the same LES input as ERA5, without any source specific code outside the source.
"""

import numpy as np
import pytest
import xarray as xr

import ls2d
import synthetic_plev as plv
from ls2d.forcing.vertical import to_terrain_following

z = np.arange(10, 5000, 50).astype(float)


@pytest.fixture
def plev(clean_registry):
    return plv.make_source()


def les(settings, n_av=1, method='2nd'):
    raw = ls2d.read(settings)
    return ls2d.get_les_input(ls2d.calculate_forcings(raw, n_av=n_av, method=method), z)


def test_sources_satisfy_standard(plev):
    ls2d.era5.validate()
    plev.validate()


def test_default_outputs(plev):
    era5 = set(ls2d.era5.default_outputs())
    other = set(plev.default_outputs())
    assert era5 == set(ls2d.les_outputs())
    assert other < era5
    # No ozone, SST, roughness, or (HTESSEL) land surface in the pressure level source.
    assert era5 - other == {
        'o3', 'o3_lay', 'sst', 'z0m', 'z0h', 'lai_low_veg', 'lai_high_veg', 'c_low_veg', 'c_high_veg',
        't_soil', 'theta_soil', 'type_soil', 'type_low_veg', 'type_high_veg', 'root_frac_low_veg',
        'root_frac_high_veg',
    }  # fmt: skip
    # Only what is needed is read.
    assert {f.key for f in plev.required_fields(['ps', 'ts'])} == {'sfc:sp', 'sfc:skt'}


@pytest.mark.parametrize('method', ['2nd', '4th'])
def test_les_input_is_source_agnostic(settings, plev, method):
    a = les(settings, method=method)
    b = les(dict(settings, source=plv.NAME), method=method)

    assert a.attrs['ls2d_source'] == 'era5' and b.attrs['ls2d_source'] == plv.NAME

    # Same definition (dims, units, long names) for every variable both provide.
    assert set(b.data_vars) <= set(a.data_vars)
    for v in b.data_vars:
        assert a[v].dims == b[v].dims, v
        assert a[v].attrs == b[v].attrs, v
    for c in ('z', 'time'):
        np.testing.assert_array_equal(a[c], b[c])

    # Same values, up to the different vertical resolution (37 pressure levels vs. 137 model levels).
    for v in b.data_vars:
        if 'lay' in b[v].dims or 'lev' in b[v].dims:
            continue  # On the native levels of each source.
        scale = float(np.abs(a[v]).max()) or 1.0
        np.testing.assert_allclose(b[v], a[v], rtol=0, atol=0.03 * scale, err_msg=v)

    # Radiation profiles: same physical profile, on the native levels of each source. Compared below
    # 50 hPa only: above, the pressure levels (10, 7, 5, 3, 2, 1 hPa) are too sparse for linear interpolation.
    below = b.p_lay[0] > 5000
    for v in ('t_lay', 'p_lay', 'h2o_lay'):
        ref = np.interp(b.z_lay[0], a.z_lay[0], a[v][0])
        scale = float(np.abs(ref[below]).max())
        np.testing.assert_allclose(b[v][0][below], ref[below], rtol=0, atol=0.03 * scale, err_msg=v)


def test_scheme_specific_outputs_are_marked(settings):
    a = les(settings)
    assert a.type_soil.attrs['land_surface_scheme'] == 'HTESSEL'
    assert 'land_surface_scheme' not in a.thl.attrs


def test_standard_units_are_enforced(plev):
    with pytest.raises(ValueError, match='standard units'):
        plev.quantity('T', requires=('pl:t',), units='degC', replace=True)(lambda t, ctx: t - 273.15)


def test_standard_dims_are_enforced(settings, plev):
    @plev.quantity('ps', requires=('sfc:sp',), replace=True)
    def ps(sp, ctx):
        return sp.isel(time=0)  # Wrong: misses `time`.

    with pytest.raises(ValueError, match='"ps" has dims'):
        ls2d.calculate_forcings(ls2d.read(dict(settings, source=plv.NAME)), n_av=1)


def test_validate_lists_missing_quantities(clean_registry):
    src = ls2d.Source('incomplete')
    src.field('t', 'pl')

    @src.quantity('T', requires=('pl:t',))
    def T(t, ctx):
        return t

    with pytest.raises(ValueError) as e:
        src.validate()
    msg = str(e.value)
    for name in ('"qv"', '"ps"', '"thl"', '"p"'):
        assert name in msg
    assert '"T"' not in msg


def test_unknown_source():
    with pytest.raises(KeyError, match='Unknown source "gfs_does_not_exist"'):
        ls2d.get_source('gfs_does_not_exist')


def test_terrain_following_ignores_levels_below_ground():
    p_levels = np.array([100000.0, 92500, 85000, 70000, 50000, 30000, 10000])
    ps_value = 95000.0
    # Linear in ln(p) above the surface, garbage below.
    values = np.where(p_levels <= ps_value, 10 * np.log(p_levels), 1e6)

    da = xr.DataArray(
        values[None, :, None, None] * np.ones((2, 1, 2, 3)),
        dims=('time', 'pressure_level', 'latitude', 'longitude'),
        coords=dict(pressure_level=p_levels),
    )
    ps = xr.DataArray(np.full((2, 2, 3), ps_value), dims=('time', 'latitude', 'longitude'))
    sigma = np.array([1.0, 0.9, 0.5, 0.2])

    out = to_terrain_following(da, ps, sigma)
    assert out.dims == ('time', 'level', 'latitude', 'longitude')
    np.testing.assert_allclose(out[0, :, 0, 0], 10 * np.log(sigma * ps_value))


def test_half_level_fallback(settings, plev):
    raw = ls2d.read(dict(settings, source=plv.NAME))
    f = ls2d.compute_fields(raw, ['p', 'ph', 'ps'])
    np.testing.assert_array_equal(f.ph.isel(level_half=0), f.ps)
    assert (f.ph.diff('level_half') < 0).all()
    assert f.ph.sizes['level_half'] == f.p.sizes['level'] + 1
