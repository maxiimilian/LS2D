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

# Python modules
import os

# Third party modules
import xarray as xr

# LS2D modules
from ls2d.core.logger import logger
import ls2d.ecmwf.era_tools as era_tools
from ls2d.ecmwf.patch_cds_ads import patch_netcdf
from ls2d.forcing.pipeline import calculate_forcings
from ls2d.forcing.source import get_source
from ls2d.forcing.les import get_les_input, les_outputs
from ls2d.forcing.raw import standardize, missing_fields, group_by_levtype

# ERA5 level type -> file type of the CDS/MARS downloads.
ftypes = {'ml': 'model_an', 'pl': 'pressure_an', 'sfc': 'surface_an'}


def _open_days(files, flds):
    """
    Open the daily NetCDF files of one file type, check the content, and merge them in time.
    """
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        msg = 'Missing ERA5 file(s): {}. Run `ls2d.download_era5()` first.'.format(', '.join(missing))
        logger.error(msg)
        raise FileNotFoundError(msg)

    datasets = []
    for f in files:
        ds = xr.open_dataset(f)

        # Fallback in case someone has unpatched NetCDF files from the new CDS (>09/2024).
        # The patching is normally done automatically after downloading the files.
        if 'valid_time' in ds.dims:
            ds.close()
            patch_netcdf(f)
            ds = xr.open_dataset(f)

        missing = missing_fields(ds, flds)
        if missing:
            msg = (
                f'ERA5 file "{f}" does not contain the field(s) {[m.key for m in missing]}. '
                f'Re-download with `ls2d.download_era5()`, which updates files with missing fields.'
            )
            logger.error(msg)
            raise KeyError(msg)

        datasets.append(ds[[fld.name for fld in flds]])

    return xr.concat(datasets, dim='time') if len(datasets) > 1 else datasets[0]


def read_era5(settings, outputs=None, fields=None):
    """
    Read the ERA5 files downloaded by `ls2d.download_era5()`, and return the raw ERA5 dataset
    (see `ls2d.forcing.raw`) for the period `start_date` to `end_date`.

    Arguments:
        settings : dict
            Dictionary with keys `central_lat`, `central_lon`, `era5_path`,
            `case_name`, `start_date`, `end_date`, and optionally `data_source`.
            Data source 'ARCO' reads the files from `ls2d.download_era5_arco()`.
        outputs : list of str, optional
            LES outputs and/or column variables (see `ls2d.les_outputs()`);
            only the ERA5 fields needed for these are read. Default: all LES outputs.
        fields : list of `Era5Field`, optional
            Read exactly these ERA5 fields (overrides `outputs`).

    Returns:
        xarray.Dataset
    """

    if settings.get('data_source', 'CDS') == 'ARCO':
        from ls2d.google.read_era5_arco import read_era5_arco_raw

        return read_era5_arco_raw(settings, outputs, fields)

    fields = get_source('era5').required_fields(outputs) if fields is None else fields

    start = era_tools.lower_to_hour(settings['start_date'])
    end = era_tools.lower_to_hour(settings['end_date'])
    logger.info(f'Reading ERA5 (CDS/MARS) for period: {start} to {end}')

    an_dates = era_tools.get_required_analysis(start, end)

    datasets = {}
    for levtype, flds in group_by_levtype(fields).items():
        files = [
            era_tools.era5_file_path(
                d.year, d.month, d.day, settings['era5_path'], settings['case_name'], ftypes[levtype], False
            )
            for d in an_dates
        ]
        datasets[levtype] = _open_days(files, flds).sel(time=slice(start, end))

    attrs = dict(
        central_lat=settings['central_lat'],
        central_lon=settings['central_lon'],
        source=f'ERA5 ({settings.get("data_source", "CDS")})',
        ls2d_source='era5',
    )
    return standardize(datasets, fields, attrs, pressure_level_dim='level').load()


class Read_era5:
    """
    Backwards compatible wrapper around the stateless functions
    `ls2d.read_era5()`, `ls2d.calculate_forcings()`, and `ls2d.get_les_input()`:

        era = ls2d.Read_era5(settings)
        era.calculate_forcings(n_av=1, method='2nd')
        les_input = era.get_les_input(z)

    The intermediate results are available as `era.raw` (raw ERA5 dataset)
    and `era.column` (column dataset after `calculate_forcings()`).
    """

    def __init__(self, settings, outputs=None):
        self.settings = dict(settings)
        self.outputs = outputs
        self.raw = read_era5(self.settings, outputs)
        self.column = None

    def calculate_forcings(self, n_av=0, method='4th'):
        """
        Calculate the advective tendencies, geostrophic wind, et cetera.
        """
        self.column = calculate_forcings(self.raw, n_av=n_av, method=method, outputs=self.outputs)
        return self.column

    def get_les_input(self, z):
        """
        Interpolate variables required for LES onto model grid,
        and return xarray.Dataset with all possible LES input.
        """
        if self.column is None:
            msg = 'Call `calculate_forcings()` before `get_les_input()`.'
            logger.error(msg)
            raise RuntimeError(msg)

        # Column-only names in `outputs` (e.g. `dtu_total`) are not LES outputs.
        outputs = None if self.outputs is None else [o for o in self.outputs if o in les_outputs()]
        return get_les_input(self.column, z, outputs=outputs or None)
