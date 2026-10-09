"""
CDS/MARS requests built from the ERA5 catalogue.
"""

import datetime

import ls2d
from ls2d.ecmwf.download_era5 import cds_request, mars_request
from ls2d.forcing.raw import group_by_levtype

date = datetime.datetime(2016, 8, 15)
settings = dict(central_lat=51.97, central_lon=4.93, area_size=1.0, era5_expver=1)

# Hard-coded requests of LS2D <= 1.1.0b1.
legacy_cds_sfc = {
    'instantaneous_moisture_flux', 'high_vegetation_cover', 'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation', 'low_vegetation_cover', 'sea_surface_temperature', 'skin_temperature',
    'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4',
    'soil_type', 'surface_pressure', 'instantaneous_surface_sensible_heat_flux', 'type_of_high_vegetation',
    'type_of_low_vegetation', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    'forecast_logarithm_of_surface_roughness_for_heat', 'forecast_surface_roughness',
}  # fmt: skip
legacy_cds_ml = set('75/76/130/131/132/133/135/203/246/247'.split('/'))
legacy_mars_sfc = set(
    '15.128/16.128/17.128/18.128/27.128/28.128/29.128/30.128/34.128/35.128/36.128/37.128/'
    '38.128/39.128/40.128/41.128/42.128/43.128/66.128/67.128/74.128/78.128/79.128/89.228/'
    '90.228/129.128/134.128/136.128/137.128/139.128/151.128/160.128/161.128/162.128/163.128/'
    '164.128/165.128/166.128/167.128/168.128/170.128/172.128/183.128/186.128/187.128/188.128/'
    '198.128/229.128/230.128/231.128/232.128/235.128/236.128/243.128/244.128/245.128'.split('/')
)


def groups(outputs=None):
    return group_by_levtype(ls2d.required_era5_fields(outputs))


def test_cds_default_matches_legacy():
    g = groups()

    name, req = cds_request('surface_an', g['sfc'], date, settings)
    assert name == 'reanalysis-era5-single-levels'
    assert set(req['variable']) == legacy_cds_sfc
    assert req['area'] == [52.97 + 0.25, 4.93 - 1 - 0.25, 50.97 - 0.25, 5.93 + 0.25]
    assert len(req['time']) == 24

    name, req = cds_request('pressure_an', g['pl'], date, settings)
    assert name == 'reanalysis-era5-pressure-levels'
    assert req['variable'] == ['geopotential']
    assert len(req['pressure_level']) == 37

    name, req = cds_request('model_an', g['ml'], date, settings)
    assert name == 'reanalysis-era5-complete'
    assert set(req['param'].split('/')) == legacy_cds_ml
    assert req['levelist'] == '/'.join(str(i) for i in range(1, 138))


def test_mars_only_requests_what_is_needed():
    g = groups()

    req = mars_request('surface_an', g['sfc'], date, settings)
    params = set(req['param'].split('/'))
    assert params < legacy_mars_sfc
    assert len(params) == 22
    assert req['levtype'] == 'sfc' and 'levelist' not in req

    req = mars_request('model_an', g['ml'], date, settings)
    assert set(req['param'].split('/')) == legacy_cds_ml
    assert req['levelist'] == '1/to/137/by/1'

    req = mars_request('pressure_an', g['pl'], date, settings)
    assert req['param'] == '129.128'
    assert req['levelist'].startswith('1/2/3/5') and req['levelist'].endswith('975/1000')


def test_subset_outputs():
    g = groups(['ps', 'ts'])
    assert set(g) == {'sfc'}
    _, req = cds_request('surface_an', g['sfc'], date, settings)
    assert set(req['variable']) == {'surface_pressure', 'skin_temperature'}
    assert mars_request('surface_an', g['sfc'], date, settings)['param'] == '134.128/235.128'
