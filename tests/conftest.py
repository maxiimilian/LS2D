import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_era5 as syn  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='session')
def era5_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('era5')
    syn.write_cds(path)
    syn.write_arco(path)
    return path


@pytest.fixture
def settings(era5_path):
    return syn.settings(era5_path)


@pytest.fixture
def clean_registry(monkeypatch):
    """
    Allow tests to register ERA5 fields, quantities, and LES outputs without affecting other tests.
    """
    from ls2d.forcing import les
    from ls2d.forcing.registry import registry

    monkeypatch.setattr(registry, '_era5', dict(registry._era5))
    monkeypatch.setattr(registry, '_quantities', dict(registry._quantities))
    monkeypatch.setattr(les, '_les_outputs', dict(les._les_outputs))
    return registry
