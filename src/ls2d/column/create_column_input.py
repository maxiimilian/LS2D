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

# LS2D modules
from ls2d.forcing.les import get_les_input, les_outputs, les_sources
from ls2d.forcing.pipeline import calculate_forcings
from ls2d.forcing.source import get_source


def create_column_input(ds, z, n_av=0, method='2nd'):
    """
    Calculate the large-scale forcings and column input for LES/SCM.
    Shortcut for `ls2d.get_les_input(ls2d.calculate_forcings(ds, n_av, method), z)`.

    Arguments:
        ds : xarray.Dataset
            Raw ERA5 dataset (`ls2d.read_era5()`), or field quantities (`ls2d.read_era5_arco()`).
        z : np.ndarray
            Full level heights LES grid (m).
        n_av : int
            Number of grid points (+/-) over which the fields and forcings are averaged.
        method : str
            '2nd' or '4th' order horizontal gradients.

    Returns:
        xarray.Dataset with the LES input.
    """

    # All LES outputs that can be computed from the content of `ds`.
    available = set(ds.data_vars)
    registry = get_source(ds.attrs.get('ls2d_source', 'era5')).registry
    outputs = [o for o in les_outputs() if all(registry.can_resolve(s, available) for s in les_sources([o]))]

    column = calculate_forcings(ds, n_av=n_av, method=method, outputs=outputs)
    return get_les_input(column, z)
