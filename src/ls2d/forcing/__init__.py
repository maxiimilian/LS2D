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

from .registry import Registry, Field, SimpleField, Quantity, core, quantity
from .context import Context, Domain
from .standard import STANDARD, StandardVar

# Populate the core registry with the source-agnostic quantities.
from . import derived as _derived, column as _column  # noqa: F401

from .les import LesVar, les_output, les_outputs, get_les_input
from .source import Source, register_source, get_source, sources
from .pipeline import (
    column_names,
    required_fields,
    required_era5_fields,
    default_outputs,
    download,
    read,
    compute_fields,
    calculate_forcings,
)
