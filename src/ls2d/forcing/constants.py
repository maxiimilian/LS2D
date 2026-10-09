#
# This file is part of LS2D.
#
# Copyright (c) 2017-2026 Wageningen University & Research
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
Physical constants used by the source-agnostic (core) quantities.
Values as in the IFS (IFS documentation part IV, chapter 12), so that ERA5 results
are identical to previous versions of LS2D.
"""

grav = 9.80665  # Gravitational acceleration (m s-2)
Rd = 287.0597  # Gas constant dry air (J kg-1 K-1)
Rv = 461.5250  # Gas constant water vapour (J kg-1 K-1)
eps = Rv / Rd - 1.0  # (-)
ep = Rd / Rv  # (-)
cpd = 1004.7090  # Specific heat dry air at constant pressure (J kg-1 K-1)
Lv = 2.5008e6  # Latent heat of vaporisation (J kg-1)
p0 = 1e5  # Reference pressure Exner function (Pa)
omega_earth = 7.2921e-5  # Angular velocity earth (s-1)

# Molar mass ratio dry air / ozone.
md_mo3 = 28.9644 / 47.9982
