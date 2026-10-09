#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Author: Bart van Stratum (WUR)
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
Catalogue of raw ERA5 fields. This is the single source of truth for the
CDS, MARS, and Google ARCO download requests and for the readers.

To make a new ERA5 variable available, add one line here (or call
`ls2d.era5_field(...)` from your own script), and use it as `{levtype}:{name}`
in the `requires` of a quantity.
"""

from ls2d.forcing.registry import era5_field

# fmt: off
# Model level analysis. `param` as used by the CDS "complete" and MARS requests.
era5_field('t',    'ml', '130', arco='temperature',                         units='K',       long_name='temperature')
era5_field('u',    'ml', '131', arco='u_component_of_wind',                 units='m s-1',   long_name='u component of wind')
era5_field('v',    'ml', '132', arco='v_component_of_wind',                 units='m s-1',   long_name='v component of wind')
era5_field('w',    'ml', '135', arco='vertical_velocity',                   units='Pa s-1',  long_name='vertical velocity')
era5_field('q',    'ml', '133', arco='specific_humidity',                   units='kg kg-1', long_name='specific humidity')
era5_field('clwc', 'ml', '246', arco='specific_cloud_liquid_water_content', units='kg kg-1', long_name='specific cloud liquid water content')
era5_field('ciwc', 'ml', '247', arco='specific_cloud_ice_water_content',    units='kg kg-1', long_name='specific cloud ice water content')
era5_field('crwc', 'ml', '75',  arco='specific_rain_water_content',         units='kg kg-1', long_name='specific rain water content')
era5_field('cswc', 'ml', '76',  arco='specific_snow_water_content',         units='kg kg-1', long_name='specific snow water content')
era5_field('o3',   'ml', '203', arco='ozone_mass_mixing_ratio',             units='kg kg-1', long_name='ozone mass mixing ratio')

# Pressure level analysis.
era5_field('z', 'pl', '129.128', cds='geopotential', arco='geopotential', units='m2 s-2', long_name='geopotential')

# Surface / single level analysis.
era5_field('sp',     'sfc', '134.128', cds='surface_pressure',                                 arco='surface_pressure',                                 units='Pa',          long_name='surface pressure')
era5_field('skt',    'sfc', '235.128', cds='skin_temperature',                                 arco='skin_temperature',                                 units='K',           long_name='skin temperature')
era5_field('sst',    'sfc', '34.128',  cds='sea_surface_temperature',                          arco='sea_surface_temperature',                          units='K',           long_name='sea surface temperature')
era5_field('ishf',   'sfc', '231.128', cds='instantaneous_surface_sensible_heat_flux',         arco='instantaneous_surface_sensible_heat_flux',         units='W m-2',       long_name='instantaneous surface sensible heat flux')
era5_field('ie',     'sfc', '232.128', cds='instantaneous_moisture_flux',                      arco='instantaneous_moisture_flux',                      units='kg m-2 s-1',  long_name='instantaneous moisture flux')
era5_field('fsr',    'sfc', '244.128', cds='forecast_surface_roughness',                       arco='forecast_surface_roughness',                       units='m',           long_name='forecast surface roughness')
era5_field('flsr',   'sfc', '245.128', cds='forecast_logarithm_of_surface_roughness_for_heat', arco='forecast_logarithm_of_surface_roughness_for_heat', units='~',           long_name='forecast logarithm of surface roughness for heat')
era5_field('slt',    'sfc', '43.128',  cds='soil_type',                                        arco='soil_type',                                        units='~',           long_name='soil type')
era5_field('tvl',    'sfc', '29.128',  cds='type_of_low_vegetation',                           arco='type_of_low_vegetation',                           units='~',           long_name='type of low vegetation')
era5_field('tvh',    'sfc', '30.128',  cds='type_of_high_vegetation',                          arco='type_of_high_vegetation',                          units='~',           long_name='type of high vegetation')
era5_field('lai_lv', 'sfc', '66.128',  cds='leaf_area_index_low_vegetation',                   arco='leaf_area_index_low_vegetation',                   units='m2 m-2',      long_name='leaf area index, low vegetation')
era5_field('lai_hv', 'sfc', '67.128',  cds='leaf_area_index_high_vegetation',                  arco='leaf_area_index_high_vegetation',                  units='m2 m-2',      long_name='leaf area index, high vegetation')
era5_field('cvl',    'sfc', '27.128',  cds='low_vegetation_cover',                             arco='low_vegetation_cover',                             units='(0 - 1)',     long_name='low vegetation cover')
era5_field('cvh',    'sfc', '28.128',  cds='high_vegetation_cover',                            arco='high_vegetation_cover',                            units='(0 - 1)',     long_name='high vegetation cover')
for _i, _param in enumerate(('139.128', '170.128', '183.128', '236.128'), start=1):
    era5_field(f'stl{_i}', 'sfc', _param, cds=f'soil_temperature_level_{_i}', arco=f'soil_temperature_level_{_i}', units='K', long_name=f'soil temperature level {_i}')
for _i, _param in enumerate(('39.128', '40.128', '41.128', '42.128'), start=1):
    era5_field(f'swvl{_i}', 'sfc', _param, cds=f'volumetric_soil_water_layer_{_i}', arco=f'volumetric_soil_water_layer_{_i}', units='m3 m-3', long_name=f'volumetric soil water layer {_i}')
# fmt: on
