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
import os

# Third party modules
import xarray as xr

# LS2D modules
import ls2d.ecmwf.era_tools as era_tools
from ls2d.core.logger import logger
from ls2d.forcing.pipeline import compute_fields
from ls2d.forcing.source import get_source
from ls2d.forcing.raw import standardize, missing_fields, group_by_levtype


def read_era5_arco_raw(settings, outputs=None, fields=None):
    """
    Read ERA5 files downloaded by `download_era5_arco()`, and
    return the raw ERA5 dataset (see `ls2d.forcing.raw`).

    Arguments:
        settings : dict
            Dictionary with keys `central_lat`, `central_lon`, `era5_path`,
            `case_name`, `start_date`, and `end_date`.
        outputs : list of str, optional
            LES outputs and/or column variables; only the ERA5 fields needed for these are read.
            Default: all LES outputs.
        fields : list of `Era5Field`, optional
            Read exactly these ERA5 fields (overrides `outputs`).
    """

    fields = get_source('era5').required_fields(outputs) if fields is None else fields

    start = era_tools.lower_to_hour(settings['start_date'])
    end = era_tools.lower_to_hour(settings['end_date'])

    logger.info(f'Reading ERA5 (Google ARCO) for period: {start} to {end}')

    files = [
        era_tools.era5_file_path(
            d.year, d.month, d.day, settings['era5_path'], settings['case_name'], 'era5_arco', False
        )
        for d in era_tools.get_required_analysis(start, end)
    ]

    datasets = []
    for f in files:
        if not os.path.exists(f):
            msg = f'File "{f}" does not exist. Run `ls2d.download_era5_arco()` first.'
            logger.error(msg)
            raise FileNotFoundError(msg)

        ds = xr.open_dataset(f)
        missing = missing_fields(ds, fields)
        if missing:
            msg = (
                f'ERA5 file "{f}" does not contain the field(s) {[m.key for m in missing]}. '
                f'Re-download with `ls2d.download_era5_arco()`, which updates files with missing fields.'
            )
            logger.error(msg)
            raise KeyError(msg)
        datasets.append(ds[[fld.name for fld in fields]])

    era = xr.concat(datasets, dim='time').sel(time=slice(start, end))

    attrs = dict(
        central_lat=settings['central_lat'],
        central_lon=settings['central_lon'],
        source='ERA5 (Google ARCO)',
        ls2d_source='era5',
    )
    return standardize({levtype: era for levtype in group_by_levtype(fields)}, fields, attrs).load()


def read_era5_arco(settings, outputs=None):
    """
    Read ERA5 files downloaded by `download_era5_arco()`, and return the
    field quantities (thl, qt, p, z, ...) on the 3D ERA5 grid.

    Equivalent to `ls2d.compute_fields(ls2d.read_era5(settings | {'data_source': 'ARCO'}))`.
    """
    return compute_fields(read_era5_arco_raw(settings, outputs))
