#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
# Authors: Maximilian Pierzyna (TU Delft), Claude (Anthropic)
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
LES/SCM input from a NOAA GFS forecast. Requires `eccodes` (`pip install ls2d[gfs]`).

GFS 0.25 degree forecasts are downloaded from the public NOAA archive on AWS (no account needed;
recent cycles appear ~4 h after initialisation, the archive goes back to early 2021). Only the
required GRIB messages are downloaded (~300 MB per forecast hour, ~10 s), and saved cropped
to the domain (~0.5 MB per forecast hour).
"""

# Python modules
from datetime import datetime, timedelta, timezone

# Third-party modules
import numpy as np
import ls2d

# Latest cycle that is certainly complete: 00 UTC of yesterday.
cycle = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

settings = {
    'source': 'gfs',
    'central_lat': 51.97,
    'central_lon': 4.93,
    'area_size': 1,  # Download +/- 1 degree around the central lat/lon.
    'case_name': 'cabauw',
    'gfs_path': '/home/scratch1/meteo_data/LS2D_GFS/',
    'gfs_cycle': cycle,  # Optional: default = last cycle at or before `start_date`.
    'start_date': cycle + timedelta(hours=6),  # Valid times of the forecast. Hourly output up to +120 h,
    'end_date': cycle + timedelta(hours=30),  # 3-hourly after (up to +384 h).
    'data_source': 'AWS',  # Or 'GCS' (Google Cloud mirror).
}

# Download (only what is needed for `outputs`; default all LES variables GFS can provide):
ls2d.download(settings)

# Read, calculate forcings, and interpolate to the LES grid; identical to ERA5:
raw = ls2d.read(settings)
column = ls2d.calculate_forcings(raw, n_av=1, method='2nd')
les_input = ls2d.get_les_input(column, z=np.arange(10, 5000, 20.0))

les_input.to_netcdf('ls2d_gfs.nc')
