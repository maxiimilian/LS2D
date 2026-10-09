import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_era5 as syn  # noqa: E402
import synthetic_plev as plv  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='session')
def era5_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('era5')
    syn.write_cds(path)
    syn.write_arco(path)
    plv.write_plev(path)
    return path


@pytest.fixture
def settings(era5_path):
    return syn.settings(era5_path)


@pytest.fixture
def clean_registry(monkeypatch):
    """
    Allow tests to register ERA5 fields, quantities, and LES outputs without affecting other tests.
    """
    import ls2d
    from ls2d.forcing import les, source

    for reg in (ls2d.core, ls2d.era5.registry):
        monkeypatch.setattr(reg, '_fields', dict(reg._fields))
        monkeypatch.setattr(reg, '_quantities', dict(reg._quantities))
    monkeypatch.setattr(les, '_les_outputs', dict(les._les_outputs))
    monkeypatch.setattr(source, '_sources', dict(source._sources))
    return ls2d.era5.registry
