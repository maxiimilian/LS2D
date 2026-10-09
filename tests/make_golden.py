"""
Generate the golden reference outputs in `tests/data/` with the *pre-refactor* code
(LS2D/LS2D@develop_arco, 1941dd7). Kept for provenance; it only runs against that version:

    git checkout 1941dd7 -- src && python tests/make_golden.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_era5 as syn  # noqa: E402

import ls2d  # noqa: E402

out = os.path.join(os.path.dirname(__file__), 'data')
z = np.arange(10, 5000, 50).astype(float)
enc = lambda ds: {v: {'zlib': True} for v in ds.data_vars}  # noqa: E731

with tempfile.TemporaryDirectory() as tmp:
    syn.write_cds(tmp)
    syn.write_arco(tmp)
    settings = syn.settings(tmp)

    for method in ('2nd', '4th'):
        era = ls2d.Read_era5(dict(settings))
        era.calculate_forcings(n_av=1, method=method)
        ds = era.get_les_input(z)
        ds.to_netcdf(os.path.join(out, f'golden_legacy_{method}.nc'), encoding=enc(ds))

    # Small domain (3x3 points, as with `area_size=0.25`): 2nd order gradients are one-sided at the edges.
    syn.write_cds(tmp, n=1, case='small')
    era = ls2d.Read_era5(syn.settings(tmp, case='small'))
    era.calculate_forcings(n_av=1, method='2nd')
    ds = era.get_les_input(z)
    ds.to_netcdf(os.path.join(out, 'golden_legacy_small.nc'), encoding=enc(ds))

    ds = ls2d.create_column_input(ls2d.read_era5_arco(dict(settings)), z, n_av=1)
    ds.to_netcdf(os.path.join(out, 'golden_arco.nc'), encoding=enc(ds))

print('done')
