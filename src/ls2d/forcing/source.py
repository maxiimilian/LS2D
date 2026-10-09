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

"""
Data sources. A source (ERA5, GFS, ...) has:

    - a registry (child of the `core` registry) with its raw fields, and the
      recipes that turn these into the standard quantities (`ls2d.forcing.standard`);
    - functions to download and read its raw fields. `read()` returns the raw dataset
      (see `ls2d.forcing.raw`), with attribute `ls2d_source` = the source name.

Everything after reading (`calculate_forcings()`, `get_les_input()`) is source-agnostic.

New sources are registered with `register_source()`; built-in sources are
loaded on first use from `ls2d.sources.<name>`.
"""

# Python modules
import importlib

# LS2D modules
from ls2d.core.logger import logger
from ls2d.forcing.les import les_outputs, les_sources
from ls2d.forcing.registry import Registry, SimpleField, core
from ls2d.forcing.standard import required_names

_sources = {}


class Source:
    """
    Arguments:
        name : name of the source, e.g. 'era5'.
        description : human readable description.
        field_class : class of the raw fields (subclass of `ls2d.forcing.registry.Field`).
        download : function `download(settings, outputs=None, fields=None)`, optional.
        read : function `read(settings, outputs=None, fields=None)` -> raw dataset.
    """

    def __init__(self, name, description='', field_class=SimpleField, download=None, read=None):
        self.name = name
        self.description = description
        self.field_class = field_class
        self.registry = Registry(name, parent=core)
        self._download = download
        self._read = read

    def __repr__(self):
        return f'Source({self.name!r})'

    #
    # Registration.
    #
    def field(self, *args, replace=False, **kwargs):
        """
        Register a raw field; arguments as `field_class`.
        """
        return self.registry.add_field(self.field_class(*args, **kwargs), replace)

    def quantity(self, *args, **kwargs):
        """
        Decorator to register a quantity specific for this source; see `Registry.quantity()`.
        """
        return self.registry.quantity(*args, **kwargs)

    #
    # Queries.
    #
    def can_provide(self, name):
        return self.registry.can_provide(name)

    def default_outputs(self):
        """
        All LES outputs this source can provide.
        """
        return [o for o in les_outputs() if all(self.can_provide(s) for s in les_sources([o]))]

    def column_names(self, outputs=None):
        """
        Column variables needed for `outputs` (LES output and/or column variable names).
        Default: all LES outputs this source can provide.
        """
        if outputs is None:
            outputs = self.default_outputs()

        available = set(self.registry.column_names())
        names = []
        for o in outputs:
            if o in les_outputs():
                names.extend(les_sources([o]))
            elif o in available:
                names.append(o)
            elif o in self.registry:
                raise ValueError(
                    f'"{o}" is a field quantity or raw field without `reduce`; it can not be used as column output.'
                )
            else:
                raise KeyError(self.registry._unknown(o))
        return list(dict.fromkeys(names))

    def required_fields(self, outputs=None):
        """
        Raw fields needed to compute `outputs` (default: all LES outputs this source can provide).
        """
        return self.registry.required_fields(self.column_names(outputs))

    def validate(self):
        """
        Check that the source can provide all required standard quantities.
        Raises ValueError with the list of problems.
        """
        problems = []
        for name in required_names():
            try:
                fields = self.registry.required_fields([name])
            except (KeyError, ValueError) as e:
                problems.append(f'"{name}": {e}')
                continue
            missing = [f.key for f in fields if f.key not in {g.key for g in self.registry.fields()}]
            if missing:
                problems.append(f'"{name}": missing raw fields {missing}')
        if problems:
            msg = f'Source "{self.name}" does not satisfy the standard:\n  ' + '\n  '.join(problems)
            logger.error(msg)
            raise ValueError(msg)

    #
    # I/O.
    #
    def download(self, settings, outputs=None, fields=None, **kwargs):
        if self._download is None:
            raise NotImplementedError(f'Source "{self.name}" has no download function')
        return self._download(settings, outputs=outputs, fields=fields, **kwargs)

    def read(self, settings, outputs=None, fields=None):
        if self._read is None:
            raise NotImplementedError(f'Source "{self.name}" has no read function')
        raw = self._read(settings, outputs=outputs, fields=fields)
        if raw.attrs.get('ls2d_source') != self.name:
            raise ValueError(f'Reader of source "{self.name}" should set attribute `ls2d_source="{self.name}"`')
        return raw


def register_source(source, replace=False):
    if source.name in _sources and not replace:
        raise KeyError(f'Source "{source.name}" already registered')
    _sources[source.name] = source
    return source


def get_source(source):
    """
    Source by name (or the source itself). Built-in sources are imported from `ls2d.sources.<name>`.
    """
    if isinstance(source, Source):
        return source
    if source not in _sources:
        module = f'ls2d.sources.{source}'
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module:
                raise
    if source not in _sources:
        raise KeyError(f'Unknown source "{source}". Registered: {sorted(_sources)}')
    return _sources[source]


def sources():
    return dict(_sources)
