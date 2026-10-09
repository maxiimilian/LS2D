#
# This file is part of LS2D.
#
# Copyright (c) 2017-2024 Wageningen University & Research
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
import subprocess as sp
import sys, os
import dill as pickle
import requests

# Third party modules
import numpy as np

# LS2D modules
import ls2d.ecmwf.era_tools as era_tools
from ls2d.core.logger import logger
from ls2d.ecmwf.patch_cds_ads import patch_netcdf, regrid_netcdf
from ls2d.forcing.raw import group_by_levtype, fields_to_download
from ls2d.forcing.source import get_source

# Yikes, but necessary (?) if you want to use
# MARS downloads without the Python CDS api installed?
try:
    import cdsapi
except ImportError:
    cdsapi = None


def _retrieve_from_MARS(request, settings, nc_dir, nc_file, qos):
    """
    Retrieve file from MARS
    """

    def execute(task):
        sp.call(task, shell=True, executable='/bin/bash')

    clean_name = nc_file[:-3]
    mars_req = '{}.mars'.format(clean_name)
    grib_file = '{}.grib'.format(clean_name)
    slurm_job = '{}.slurm'.format(clean_name)

    # Wall clock limit
    wc_lim = '03:00:00' if qos == 'express' else '06:00:00'

    # Create MARS request
    f = open(mars_req, 'w')
    f.write('retrieve,\n')
    for key, value in request.items():
        f.write('{}={},\n'.format(key, value))
    f.write('target="{}"\n'.format(grib_file))
    f.close()

    # Create SLURM job file
    date = settings['date']
    ftype = settings['ftype'].split('_')
    jobname = '{0:04d}{1:02d}{2:02d}{3:}{4:}'.format(date.year, date.month, date.day, ftype[1], ftype[0])

    f = open(slurm_job, 'w')
    f.write('#!/bin/ksh\n')
    f.write('#SBATCH --qos={}\n'.format(qos))
    f.write('#SBATCH --job-name={}\n'.format(jobname))
    f.write('#SBATCH --output={}.%N.%j.out\n'.format(slurm_job))
    f.write('#SBATCH --error={}.%N.%j.err\n'.format(slurm_job))
    f.write('#SBATCH --chdir={}\n'.format(nc_dir))
    f.write('#SBATCH --time={}\n\n'.format(wc_lim))

    f.write('mars {}\n'.format(mars_req))
    f.write('module load ecmwf-toolbox \n')
    f.write('grib_to_netcdf -o {} {}'.format(nc_file, grib_file))
    f.close()

    # Submit job
    execute('sbatch {}'.format(slurm_job))


# ERA5 level type <-> file type of the downloads.
ftypes = {'ml': 'model_an', 'pl': 'pressure_an', 'sfc': 'surface_an'}
levtypes = {v: k for k, v in ftypes.items()}

pressure_levels = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350,
    400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]  # fmt: skip


def _area(settings):
    """
    Bounds of domain: north, west, south, east.
    """
    size = settings['area_size']
    lat, lon = settings['central_lat'], settings['central_lon']
    return lat + size, lon - size, lat - size, lon + size


def cds_request(ftype, fields, date, settings):
    """
    Build CDS request for one day and file type.

    Returns:
        (CDS dataset name, request dictionary)
    """
    lat_n, lon_w, lat_s, lon_e = _area(settings)

    if ftype in ('pressure_an', 'surface_an'):
        missing = [f.key for f in fields if f.cds is None]
        if missing:
            raise ValueError(f'ERA5 field(s) {missing} have no CDS name in the registry.')

        # Add +/- 1 grid point to pressure and surface files, required for interpolations.
        pad = 0.25
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'year': '{0:04d}'.format(date.year),
            'month': '{0:02d}'.format(date.month),
            'day': '{0:02d}'.format(date.day),
            'time': ['{0:02d}:00'.format(i) for i in range(24)],
            'area': [lat_n + pad, lon_w - pad, lat_s - pad, lon_e + pad],
            'variable': [f.cds for f in fields],
        }

        if ftype == 'pressure_an':
            request['pressure_level'] = [str(p) for p in pressure_levels]
            return 'reanalysis-era5-pressure-levels', request
        return 'reanalysis-era5-single-levels', request

    elif ftype == 'model_an':
        # Model level analysis, stored in tape archive, so downloads are VERY slow :-(
        request = {
            'class': 'ea',
            'date': '{0:04d}-{1:02d}-{2:02d}'.format(date.year, date.month, date.day),
            'expver': '1',
            'levelist': '/'.join(list(np.arange(1, 138).astype(str))),
            'levtype': 'ml',
            'param': '/'.join(f.param for f in fields),
            'stream': 'oper',
            'time': '/'.join(['{0:02d}:00:00'.format(i) for i in range(24)]),
            'type': 'an',
            'area': '{}/{}/{}/{}'.format(lat_n, lon_w, lat_s, lon_e),
            'grid': '0.25/0.25',
            'format': 'netcdf',
        }
        return 'reanalysis-era5-complete', request

    raise ValueError(f'Unknown file type "{ftype}"')


def mars_request(ftype, fields, date, settings):
    """
    Build MARS request for one day and file type.
    """
    lat_n, lon_w, lat_s, lon_e = _area(settings)

    request = {
        'class': 'ea',
        'expver': '{}'.format(settings['era5_expver']),
        'stream': 'oper',
        'date': '{0:04d}-{1:02d}-{2:02d}'.format(date.year, date.month, date.day),
        'area': '{}/{}/{}/{}'.format(lat_n, lon_w, lat_s, lon_e),
        'grid': '0.25/0.25',
        'format': 'netcdf',
        'levtype': levtypes[ftype],
        'type': 'an',
        'time': '0/to/23/by/1',
        'param': '/'.join(f.param for f in fields),
    }

    if ftype == 'model_an':
        request['levelist'] = '1/to/137/by/1'
    elif ftype == 'pressure_an':
        request['levelist'] = '/'.join(str(p) for p in pressure_levels)

    return request


def _download_era5_file(settings):
    """
    Download (CDS) or submit (MARS) the ERA5 analysis on surface, model or pressure levels for one day.

    Arguments:
        settings : dictionary
            LS2D settings, plus keys:
                date : datetime object with date to download
                ftype : file type (in: [model_an, pressure_an, surface_an])
                fields : list of `Era5Field` to download
    """

    logger.info(f'Downloading ERA5 ({settings["data_source"]}) for {settings["date"]:%Y-%m-%d} - {settings["ftype"]}')

    # Keep track of CDS downloads which are finished:
    finished = False

    # Output file name
    nc_dir, nc_file = era_tools.era5_file_path(
        settings['date'].year,
        settings['date'].month,
        settings['date'].day,
        settings['era5_path'],
        settings['case_name'],
        settings['ftype'],
    )

    # Write CDS API prints to log file (NetCDF file path/name appended with .out/.err)
    if settings['write_log']:
        out_file = '{}.out'.format(nc_file[:-3])
        err_file = '{}.err'.format(nc_file[:-3])
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(out_file, 'w')
        sys.stderr = open(err_file, 'w')

    # Switch between CDS and MARS downloads
    if settings['data_source'] == 'CDS':
        # Check if pickle with previous request is available.
        # If so, try to download NetCDF file, if not, submit new request
        pickle_file = '{}.pickle'.format(nc_file[:-3])

        if os.path.isfile(pickle_file):
            logger.info('Found previous CDS request!')

            with open(pickle_file, 'rb') as f:
                cds_request_obj = pickle.load(f)

                try:
                    cds_request_obj.update()
                except requests.exceptions.HTTPError:
                    logger.error('CDS request is no longer available online!')
                    msg = 'To continue, delete the previous request: {}'.format(pickle_file)
                    logger.error(msg)
                    raise RuntimeError(msg)

                state = cds_request_obj.reply['state']

                if state == 'completed':
                    logger.info('Request finished, downloading NetCDF file')

                    cds_request_obj.download(nc_file)
                    f.close()
                    os.remove(pickle_file)

                    # Patch NetCDF file, to make the (+/-) identical to the old CDS
                    # files, and files retrieved from MARS.
                    patch_netcdf(nc_file)

                    # Interpolate to requested grid, to stay in line with downloads before March 2026.
                    if settings['ftype'] in ('surface_an', 'pressure_an'):
                        regrid_netcdf(nc_file, settings['central_lon'], settings['central_lat'], resolution=0.25)

                    finished = True

                elif state in ('queued', 'accepted', 'running'):
                    logger.info('Request not finished, current status = "{}"'.format(state))

                else:
                    logger.error('Request failed, status = "{}"'.format(state))
                    logger.error('Error message = {}'.format(cds_request_obj.reply['error'].get('message')))
                    logger.error('Error reason = {}'.format(cds_request_obj.reply['error'].get('reason')))

        else:
            logger.info('No previous CDS request, submitting new one')

            server = cdsapi.Client(wait_until_complete=False, delete=False)
            dataset, request = cds_request(settings['ftype'], settings['fields'], settings['date'], settings)
            cds_request_obj = server.retrieve(dataset, request)

            # Save pickle for later processing/download
            with open(pickle_file, 'wb') as f:
                pickle.dump(cds_request_obj, f)

    elif settings['data_source'] == 'MARS':
        request = mars_request(settings['ftype'], settings['fields'], settings['date'], settings)
        _retrieve_from_MARS(request, settings, nc_dir, nc_file, qos='nf')

    # Restore printing to screen
    if settings['write_log']:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return finished


def download_era5(settings, exit_when_waiting=True, outputs=None, fields=None):
    """
    Download the ERA5 fields required for an experiment between `start_date` and `end_date`,
    as 24 hour blocks (00 UTC to (including) 23 UTC).

    Only the ERA5 fields needed for `outputs` are downloaded. Existing files that miss
    one or more of the required fields are downloaded again.

    Arguments:
        settings : dict
            Dictionary with keys `central_lat`, `central_lon`, `area_size`, `era5_path`, `case_name`,
            `start_date`, `end_date`, `data_source` ('CDS', 'MARS', or 'ARCO'), `write_log`,
            and `era5_expver` (MARS only).
        exit_when_waiting : bool
            Exit Python if CDS requests are not finished yet.
        outputs : list of str, optional
            LES outputs and/or column variables (see `ls2d.les_outputs()`). Default: all LES outputs.
        fields : list of `Era5Field`, optional
            Download exactly these ERA5 fields (overrides `outputs`).
    """

    if settings['data_source'] == 'ARCO':
        from ls2d.google.download_era5_arco import download_era5_arco

        return download_era5_arco(settings, outputs=outputs, fields=fields)

    logger.info(f'Downloading ERA5 ({settings["data_source"]}) for period: {settings["start_date"]} to {settings["end_date"]}')

    # Check if output directory exists.
    if not os.path.isdir(settings['era5_path']):
        msg = 'Output directory "{}" does not exist!'.format(settings['era5_path'])
        logger.error(msg)
        raise FileNotFoundError(msg)

    if settings['data_source'] == 'CDS' and cdsapi is None:
        msg = 'CDS API is not installed. See: https://cds.climate.copernicus.eu/how-to-api'
        logger.error(msg)
        raise ImportError(msg)

    era5 = get_source('era5')
    fields = era5.required_fields(outputs) if fields is None else fields

    # Round date/time to full hours
    start = era_tools.lower_to_hour(settings['start_date'])
    end = era_tools.lower_to_hour(settings['end_date'])

    # Get list of required analysis times
    an_dates = era_tools.get_required_analysis(start, end)

    # Option to exclude download types.
    blacklist = settings.get('blacklist_download', [])

    # Loop over all required files, check if there is a local version, if not add to download queue
    download_queue = []
    for date in an_dates:
        for levtype, flds in group_by_levtype(fields).items():
            ftype = ftypes[levtype]
            if ftype in blacklist:
                continue

            era_dir, era_file = era_tools.era5_file_path(
                date.year, date.month, date.day, settings['era5_path'], settings['case_name'], ftype
            )

            if not os.path.exists(era_dir):
                logger.debug('Creating output directory {}'.format(era_dir))
                os.makedirs(era_dir)

            to_download = fields_to_download(era_file, flds, era5.registry.fields(levtype))
            if not to_download:
                logger.debug('Found {} - {} local'.format(date, ftype))
            else:
                if os.path.isfile(era_file):
                    logger.info(f'{era_file} misses required field(s), downloading again')
                download_queue.append(dict(settings, date=date, ftype=ftype, fields=to_download))

    finished = True
    for req in download_queue:
        if not _download_era5_file(req):
            finished = False

    if not finished:
        if settings['data_source'] == 'CDS':
            print(' -----------------------------------------------------------')
            print(' | One or more requests are not finished.                  |')
            print(' | For CDS request, you can monitor the progress at:       |')
            print(' | https://cds.climate.copernicus.eu/requests?tab=all      |')
            if exit_when_waiting:
                print(' | This script will stop now, you can restart it           |')
                print(' | at any time to retry, or download the results.          |')
                print(' -----------------------------------------------------------')
                sys.exit(0)
            print(' -----------------------------------------------------------')
        else:
            print(' -------------------------------------------------')
            print(' | MARS requests are submitted.                  |')
            print(' | This script will stop now, you can restart it |')
            print(' | at any time to retry.                         |')
            print(' -------------------------------------------------')

    return finished
