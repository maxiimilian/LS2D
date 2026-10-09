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

"""
Immutable context passed as last argument to every quantity function.
Replaces the state that used to live on `Read_era5` (`self.settings`, `self.i`, `self.j`, ...).
"""

# Python modules
from dataclasses import dataclass
from typing import Optional

# Third party modules
import numpy as np

# LS2D modules
import ls2d.core.finite_difference as fd
import ls2d.core.spatial_tools as spatial
from ls2d.core.logger import logger
import ls2d.forcing.constants as c

r_earth_2nd = 6.37e6  # Earth radius used by the 2nd order gradients (as in previous versions of LS2D)

METHODS = {'2nd': 1, '4th': 2}  # method -> halo size (grid points)


def format_latlon(lat, lon):
    """
    Format lat/lon as e.g. `52.00°N, 4.93°E`.
    """
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f'{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}'


@dataclass(frozen=True)
class Domain:
    """
    Averaging sub-domain around the column closest to (`central_lat`, `central_lon`):
    (1 + 2 `n_av`)² grid points, plus a halo for the horizontal gradients.

    The halo is `halo` points on each side. For the 2nd order gradients, the halo can be
    smaller if the downloaded area is small; the gradients are then one-sided at the edge.
    """

    j: int  # Index central column in full grid (latitude).
    i: int  # Index central column in full grid (longitude).
    n_av: int
    halo: int
    pad: tuple  # Halo points available (south, north, west, east).
    lat: float  # Latitude central column.
    lon: float  # Longitude central column.
    dx: float  # Grid spacing at central column (m), used by the 4th order gradients.
    dy: float
    area: str

    @classmethod
    def from_grid(cls, latitude, longitude, central_lat, central_lon, n_av, method):
        if method not in METHODS:
            raise ValueError(f'Unknown method "{method}", choose from {list(METHODS)}')
        halo = METHODS[method]

        lats = np.asarray(latitude)
        lons = np.asarray(longitude)

        i = int(np.abs(lons - central_lon).argmin())
        j = int(np.abs(lats - central_lat).argmin())

        # Available grid points outside the averaging area.
        room = (j - n_av, lats.size - 1 - j - n_av, i - n_av, lons.size - 1 - i - n_av)
        if min(room) < 0:
            msg = f'Domain too small for n_av={n_av}; download a larger area.'
            logger.error(msg)
            raise ValueError(msg)

        pad = tuple(min(halo, r) for r in room)
        if pad != (halo,) * 4:
            if method == '4th':
                msg = f'Domain too small for n_av={n_av} and method="4th"; download a larger area, or use method="2nd".'
                logger.error(msg)
                raise ValueError(msg)
            logger.warning('Averaging area touches the edge of the ERA5 domain: using one-sided gradients at the edge.')

        if pad[2] > 0 and pad[3] > 0:
            dx = spatial.dlon(lons[i - 1], lons[i + 1], lats[j]) / 2.0
        else:
            dx = np.nan
        if pad[0] > 0 and pad[1] > 0:
            dy = spatial.dlat(lats[j - 1], lats[j + 1]) / 2.0
        else:
            dy = np.nan

        dlon = (1 + 2 * n_av) * float(lons[1] - lons[0])
        dlat = (1 + 2 * n_av) * float(lats[1] - lats[0])

        return cls(j, i, n_av, halo, pad, float(lats[j]), float(lons[i]), dx, dy, f'{dlon:.2f}°×{dlat:.2f}°')

    def subset(self, ds):
        """
        Select sub-domain + halo from (full grid) `ds`.
        """
        s, n, w, e = self.pad
        return ds.isel(
            latitude=slice(self.j - self.n_av - s, self.j + self.n_av + n + 1),
            longitude=slice(self.i - self.n_av - w, self.i + self.n_av + e + 1),
        )

    #
    # Operators on fields defined on the sub-domain + halo.
    #
    def mean(self, da):
        s, _, w, _ = self.pad
        size = 2 * self.n_av + 1
        return da.isel(latitude=slice(s, s + size), longitude=slice(w, w + size)).mean(('latitude', 'longitude'))

    def nearest(self, da):
        s, _, w, _ = self.pad
        return da.isel(latitude=s + self.n_av, longitude=w + self.n_av)

    def ddx(self, da):
        if self.halo == 1:
            m_per_deg = r_earth_2nd * np.pi / 180
            return da.differentiate('longitude') / (m_per_deg * np.cos(np.deg2rad(da.latitude)))
        return _grad4c(da, 'longitude', self.dx)

    def ddy(self, da):
        if self.halo == 1:
            m_per_deg = r_earth_2nd * np.pi / 180
            return da.differentiate('latitude') / m_per_deg
        return _grad4c(da, 'latitude', self.dy)


def _grad4c(da, dim, delta):
    """
    4th order centred gradient. Halo points are NaN, but never used.
    """
    s = lambda n: da.shift({dim: n})  # noqa: E731 (value at index - n)
    return fd.grad4c(s(2), s(1), s(-1), s(-2), delta)


@dataclass(frozen=True)
class Context:
    """
    Arguments:
        central_lat, central_lon : location of the column (degrees).
        n_av : number of grid points (+/-) over which fields and forcings are averaged.
        method : '2nd' or '4th' order horizontal gradients.
        domain : averaging sub-domain; only available while computing column quantities.
    """

    central_lat: float
    central_lon: float
    n_av: int = 0
    method: str = '2nd'
    domain: Optional[Domain] = None

    @property
    def fc(self):
        """Coriolis parameter (s-1)."""
        return 2 * c.omega_earth * np.sin(np.deg2rad(self.central_lat))

    def _dom(self):
        if self.domain is None:
            raise RuntimeError('Column operators (mean, ddx, ...) are only available in column quantities.')
        return self.domain

    def mean(self, da):
        """Mean over the averaging area."""
        return self._dom().mean(da)

    def nearest(self, da):
        """Value at the central column."""
        return self._dom().nearest(da)

    def ddx(self, da):
        """Zonal gradient (per m)."""
        return self._dom().ddx(da)

    def ddy(self, da):
        """Meridional gradient (per m)."""
        return self._dom().ddy(da)

    def advec(self, da, u, v):
        """Area mean horizontal advective tendency of `da`."""
        return self.mean(-u * self.ddx(da) - v * self.ddy(da))
