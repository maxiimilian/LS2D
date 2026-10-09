#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Original author: Bart van Stratum (WUR)
# Additional authors (refactoring): Maximilian Pierzyna, Claude (Anthropic)
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

# Python modules
import datetime
import time
import os

# Third party modules
import numpy as np
import xarray as xr
import requests
import gcsfs

# LS2D modules
import ls2d.ecmwf.era_tools as era_tools
from ls2d.google.arco_tools import get_layout, read_rows
from ls2d.core.logger import logger
from ls2d.forcing.source import get_source
from ls2d.forcing.raw import fields_to_download, group_by_levtype

_bucket = 'gcp-public-data-arco-era5/ar'
_store_ml = f'{_bucket}/model-level-1h-0p25deg.zarr-v1'
_store_sl = f'{_bucket}/full_37-1h-0p25deg-chunk-1.zarr-v3'

def _open_metadata(store):
    """
    Open ARCO store for coordinates and attributes only; no data is read through xarray/zarr.
    """
    return xr.open_zarr(f'gs://{store}', chunks=None, storage_options=dict(token='anon'))


def _last_valid_date(store):
    """
    Last day with valid data in ARCO store, including ERA5T if available.
    Uses plain HTTPS, so that we can exit cleanly before opening any `gcsfs` sessions.
    """
    attrs = requests.get(f'https://storage.googleapis.com/{store}/.zattrs', timeout=30).json()
    stop = attrs.get('valid_time_stop_era5t', attrs['valid_time_stop'])
    return datetime.datetime.strptime(stop, '%Y-%m-%d')


def _clean_attrs(attrs):
    """
    Remove GRIB attributes, which are not relevant after the ARCO conversion.
    """
    return {k: v for k, v in attrs.items() if not k.startswith('GRIB')}


def _arco_fields(fields):
    """
    Check that `fields` can be downloaded from ARCO into one file.
    """
    missing = [f.key for f in fields if f.arco is None]
    if missing:
        raise ValueError(f'ERA5 field(s) {missing} are not available in Google ARCO (no `arco` name in the registry).')
    names = [f.name for f in fields]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f'ERA5 fields with the same short name {duplicates} can not be stored in one ARCO file.')
    return fields


def download_era5_arco(settings, batch_size=512, outputs=None, fields=None):
    """
    Download all required ERA5 fields for an experiment between
    `start_date` and `end_date` from the Google ARCO-ERA5 archive.

    All model level, pressure level, and surface fields are saved
    as one NetCDF file per day (00 UTC to (including) 23 UTC), on the native
    0.25 deg ERA5 grid, as: `era5_path/case_name/yyyy/mm/dd/era5_arco.nc`.

    Arguments:
        settings : dict
            Dictionary with keys:
                central_lat, central_lon : requested latitude and longitude
                area_size : download an area of lat+/-size, lon+/-size (degrees)
                era5_path : absolute or relative path to save the NetCDF data
                case_name : case name used in file path of NetCDF files
                start_date, end_date : datetime objects with start/end of experiment
        batch_size : int
            Number of concurrent HTTP range requests.
        outputs : list of str, optional
            LES outputs and/or column variables (see `ls2d.les_outputs()`); only the ERA5
            fields needed for these are downloaded. Default: all LES outputs.
        fields : list of `Era5Field`, optional
            Download exactly these ERA5 fields (overrides `outputs`).

    Existing files that miss one or more of the required fields are downloaded again.
    """

    logger.info(f'Downloading ERA5 (Google ARCO) for period: {settings["start_date"]} to {settings["end_date"]}')

    # Check if output directory exists.
    if not os.path.isdir(settings['era5_path']):
        msg = f'Output directory "{settings["era5_path"]}" does not exist!'
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Round date/time to full hours, and get list of days to download.
    start = era_tools.lower_to_hour(settings['start_date'])
    end = era_tools.lower_to_hour(settings['end_date'])
    an_dates = era_tools.get_required_analysis(start, end)

    era5 = get_source('era5')
    fields = _arco_fields(era5.required_fields(outputs) if fields is None else fields)
    candidates = [f for f in era5.registry.fields() if f.arco is not None]

    download_dates = []
    download_fields = []
    for date in an_dates:
        era_dir, era_file = era_tools.era5_file_path(
            date.year, date.month, date.day, settings['era5_path'], settings['case_name'], 'era5_arco'
        )
        to_download = fields_to_download(era_file, fields, candidates)
        if not to_download:
            logger.debug(f'Found {era_file} local')
        else:
            if os.path.isfile(era_file):
                logger.info(f'{era_file} misses required field(s), downloading again')
            download_dates.append((date, era_dir, era_file))
            download_fields += [f for f in to_download if f not in download_fields]

    groups = group_by_levtype(_arco_fields(download_fields))
    _vars_ml = {f.arco: f.name for f in groups.get('ml', [])}
    _vars_pl = {f.arco: f.name for f in groups.get('pl', [])}
    _vars_sfc = {f.arco: f.name for f in groups.get('sfc', [])}

    if len(download_dates) == 0:
        logger.info('All required ERA5 files found local, nothing to download')
        return True

    # Check if data is available for all requested days.
    last_valid = min(_last_valid_date(_store_ml), _last_valid_date(_store_sl))
    for date, _, _ in download_dates:
        if date > last_valid:
            msg = f'ERA5 data at {date:%Y-%m-%d} is not (yet) available in ARCO. Last available day: {last_valid:%Y-%m-%d}'
            logger.error(msg)
            raise ValueError(msg)

    fs = gcsfs.GCSFileSystem(token='anon')
    ds_ml = _open_metadata(_store_ml)
    ds_sl = _open_metadata(_store_sl)

    # Both stores share the same lat/lon grid and time axis.
    lats = ds_sl.latitude.values
    lons = ds_sl.longitude.values
    if not (np.array_equal(lats, ds_ml.latitude.values) and np.array_equal(lons, ds_ml.longitude.values)):
        msg = 'Model and single/pressure level ARCO stores have different grids!'
        logger.error(msg)
        raise RuntimeError(msg)

    # Select box on native grid, including one extra grid point for the gradients.
    # Latitude = contiguous rows. Longitude: full rows are read anyway, so any
    # (also 0-deg crossing) selection is free. Output longitudes are -180..180, west->east,
    # except for boxes crossing the 180 deg meridian, which keep 0..360.
    # The box is centred on the nearest grid point, with `n` points on each side.
    n = int(round(settings['area_size'] / 0.25)) + 1

    jc = np.abs(lats - settings['central_lat']).argmin()
    ilat0, nlat = jc - n, 2 * n + 1

    dlon = ((lons - settings['central_lon'] % 360 + 180) % 360) - 180
    ic = np.abs(dlon).argmin()
    ilon = (ic + np.arange(-n, n + 1)) % lons.size
    lons_out = ((lons[ilon] + 180) % 360) - 180
    if np.any(np.diff(lons_out) < 0):
        lons_out = lons[ilon]

    # Variables to read, with their chunk layout.
    layouts = [(_store_ml, name, get_layout(ds_ml, name)) for name in _vars_ml]
    layouts += [(_store_sl, name, get_layout(ds_sl, name)) for name in _vars_pl]
    layouts += [(_store_sl, name, get_layout(ds_sl, name)) for name in _vars_sfc]

    arco_ds = {**{name: ds_ml for name in _vars_ml}, **{name: ds_sl for name in {**_vars_pl, **_vars_sfc}}}
    short_names = {**_vars_ml, **_vars_pl, **_vars_sfc}

    for date, era_dir, era_file in download_dates:
        logger.info(f'Downloading ERA5 (Google ARCO) for {date:%Y-%m-%d}')

        if not os.path.exists(era_dir):
            logger.debug(f'Creating output directory {era_dir}')
            os.makedirs(era_dir)

        times = np.array([np.datetime64(date + datetime.timedelta(hours=h), 'ns') for h in range(24)])
        itimes = np.searchsorted(ds_sl.time.values, times)
        if not (np.array_equal(ds_sl.time.values[itimes], times) and np.array_equal(ds_ml.time.values[itimes], times)):
            msg = 'Requested times not found in ARCO time axis!'
            logger.error(msg)
            raise RuntimeError(msg)

        data = {name: np.empty((24, lay['nlev'], nlat, ilon.size), np.float32) for _, name, lay in layouts}

        t_start = time.perf_counter()
        for n, it in enumerate(itimes):
            rows = read_rows(fs, layouts, it, ilat0, nlat, lats.size, lons.size, batch_size)
            for name in data:
                data[name][n] = rows[name][:, :, ilon]

        # Create combined dataset and save to NetCDF.
        variables = {}
        for _, name, _ in layouts:
            da = arco_ds[name][name]
            attrs = _clean_attrs(da.attrs)
            if name in _vars_ml:
                variables[short_names[name]] = (('time', 'level', 'latitude', 'longitude'), data[name], attrs)
            elif name in _vars_pl:
                variables[short_names[name]] = (('time', 'pressure_level', 'latitude', 'longitude'), data[name], attrs)
            else:
                variables[short_names[name]] = (('time', 'latitude', 'longitude'), data[name][:, 0], attrs)

        ds = xr.Dataset(
            variables,
            coords=dict(
                time=times,
                level=('level', ds_ml.hybrid.values.astype(np.int32), dict(long_name='model level')),
                pressure_level=('pressure_level', ds_sl.level.values.astype(np.int32), ds_sl.level.attrs),
                latitude=('latitude', lats[ilat0 : ilat0 + nlat], ds_sl.latitude.attrs),
                longitude=('longitude', lons_out, ds_sl.longitude.attrs),
            ),
            attrs=dict(
                source='ERA5 from Google ARCO-ERA5 (https://github.com/google-research/arco-era5)',
                stores=f'gs://{_store_ml}, gs://{_store_sl}',
                history=f'Downloaded by (LS)2D on {datetime.datetime.now():%Y-%m-%d %H:%M}',
            ),
        )

        # Write to temporary file first, so that an interrupted download does not leave a valid looking file.
        tmp_file = f'{era_file}.tmp'
        ds.to_netcdf(tmp_file)
        os.replace(tmp_file, era_file)
        logger.info(f'Saved {era_file} in {time.perf_counter() - t_start:.0f} sec')

    return True
