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

from .registry import Registry, Era5Field, Quantity, registry, era5_field, quantity
from .context import Context, Domain

# Populate the default registry.
from . import era5_fields, fields, column  # noqa: F401

from .les import LesVar, les_output, les_outputs, get_les_input
from .pipeline import column_names, required_era5_fields, compute_fields, calculate_forcings
