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
Source-agnostic field quantities, derived from the standard quantities provided by
the data sources (see `ls2d.forcing.standard`). A source can override any of these
by registering a quantity with the same name in its own registry.

Fallbacks: a source provides either the full level pressure `p` or the half level
pressure `ph`; the other one is derived here.
"""

# Third party modules
import numpy as np

# LS2D modules
import ls2d.forcing.constants as c
from ls2d.forcing.registry import quantity
from ls2d.forcing.standard import d3h
from ls2d.forcing.vertical import half_levels_from_full, mid_levels


@quantity('Tv', requires=('T', 'qv', 'ql'))
def Tv(T, qv, ql, ctx):
    # IFS documentation part IV, eq. 12.6.
    return T * (1 + c.eps * qv - ql)


@quantity('p', requires=('ph',))
def p(ph, ctx):
    return mid_levels(ph, 'level_half', 'level').transpose('time', 'level', ...)


@quantity('ph', requires=('p', 'ps'))
def ph(p, ps, ctx):
    return half_levels_from_full(p, ps)


@quantity('zh', requires=('ph', 'Tv'))
def zh(ph, Tv, ctx):
    """
    Hydrostatic half level height above the surface (IFS documentation part III, eq. 2.20-2.21).
    """
    pv = ph.transpose(*d3h).values
    dz = -c.Rd * Tv.transpose('time', 'level', ...).values * np.log(pv[:, 1:] / pv[:, :-1]) / c.grav
    z = np.concatenate((np.zeros_like(pv[:, :1]), np.cumsum(dz, axis=1)), axis=1)
    return ph.transpose(*d3h).copy(data=z)


@quantity('z', requires=('zh',))
def z(zh, ctx):
    return mid_levels(zh, 'level_half', 'level').transpose('time', 'level', ...)


@quantity('exn', requires=('p',))
def exn(p, ctx):
    return (p / c.p0) ** (c.Rd / c.cpd)


@quantity('thl', requires=('T', 'exn', 'ql'))
def thl(T, exn, ql, ctx):
    return T / exn - c.Lv / (c.cpd * exn) * ql


@quantity('qt', requires=('qv', 'ql'))
def qt(qv, ql, ctx):
    return qv + ql


@quantity('rho', requires=('p', 'Tv'))
def rho(p, Tv, ctx):
    return p / (c.Rd * Tv)


@quantity('w', requires=('omega', 'rho'))
def w(omega, rho, ctx):
    return -omega / (rho * c.grav)


@quantity('h2o', requires=('qv', 'qt'))
def h2o(qv, qt, ctx):
    return qv / (c.ep * (1 - qt))


@quantity('rhos', requires=('ps', 'ts', 'qv'))
def rhos(ps, ts, qv, ctx):
    return ps / (c.Rd * ts * (1 + c.eps * qv.isel(level=0, drop=True)))
