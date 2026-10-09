#
# This file is part of LS2D.
#
# Copyright (c) 2017-2024 Wageningen University & Research
# Original author: Bart van Stratum (WUR)
# Additional authors (refactoring): Maximilian Pierzyna (TU Delft), Claude (Anthropic)
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

# Ban Python 2.x:
import sys

if sys.version_info.major < 3:
    raise RuntimeError('(LS)2D requires Python 3.x')

# Make packages directly available as e.g.:
# ls2d.download_era5() instead of ls2d.ecmwf.download_era5()
from ls2d.ecmwf import download_era5
from ls2d.ecmwf import download_cams
from ls2d.google import download_era5_arco, read_era5_arco
from ls2d.column import create_column_input

from ls2d.ecmwf import Read_era5, read_era5

# Stateless, registry based pipeline:
#   ls2d.read() -> ls2d.calculate_forcings() -> ls2d.get_les_input()
from ls2d.forcing import (
    core,
    quantity,
    les_output,
    les_outputs,
    Source,
    register_source,
    get_source,
    sources,
    column_names,
    required_fields,
    required_era5_fields,
    default_outputs,
    download,
    read,
    compute_fields,
    calculate_forcings,
    get_les_input,
)

# Built-in sources.
from ls2d.sources.era5 import era5

era5_field = era5.field

from ls2d.ecmwf import Read_cams

from ls2d.core import grid
