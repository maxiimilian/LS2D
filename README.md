# (LS)<sup>2</sup>D: LES and SCM - Large Scale Dynamics

[![PyPI version](https://badge.fury.io/py/ls2d.svg)](https://pypi.org/project/ls2d/)
[![Static Badge](https://img.shields.io/badge/JAMES-10.1029%2F2023MS003750-blue?link=https%3A%2F%2Fdoi.org%2F10.1029%2F2023MS003750)](https://doi.org/10.1029/2023MS003750)

(LS)<sup>2</sup>D is a Python toolkit, developed to simplify all the steps required to downscale ERA5 with doubly-periodic large-eddy simulation (LES), or single-column models (SCMs). For the retrieval of ERA data, it relies on the Copernicus Data Store (CDS), or the Meteorological Archival and Retrieval System (MARS) at ECMWF computer systems.

> [!IMPORTANT]
> In September 2024, Copernicus upgraded the Copernicus and Atmosphere Data Stores (CDS/ADS), introducing changes in the NetCDF file structure. The updated files now have different dimension names, time units, and even some reordered dimensions (!!). Unfortunately, these changes made the new NetCDF files incompatible with older versions or files extracted from MARS. Although (LS)<sup>2</sup>D briefly supported the new format, this led to several issues. Therefore, starting November 2024, (LS)<sup>2</sup>D will automatically patch new NetCDF files to ensure compatibility with the previous NetCDF file structure. On the user side, nothing should change:
> - Old NetCDF files from CDS or MARS remain compatible.
> - New NetCDF files from CDS/ADS are automatically patched after download.
> - Unpatched new NetCDF files are patched once prior to reading.
>   
> To upgrade to the new CDS and ADS, please follow the steps described [here](https://confluence.ecmwf.int/display/CKB/Please+read%3A+CDS+and+ADS+migrating+to+new+infrastructure%3A+Common+Data+Store+%28CDS%29+Engine). If (LS)<sup>2</sup>D was installed using `pip`, upgrade (LS)<sup>2</sup>D with `pip install --upgrade ls2d`.

### References

(LS)<sup>2</sup>D is described in:

B.J.H. van Stratum, C.C. van Heerwaarden, & J. Vilà-Guerau de Arellano (2023). *The benefits and challenges of downscaling a global reanalysis with doubly-periodic large-eddy simulations.* JAMES, https://doi.org/10.1029/2023MS003750

If you use (LS)<sup>2</sup>D, we kindly request citing this paper.

### Installation

If you want to use CDS to download the ERA5 data, then please start by following the steps explained at https://cds.climate.copernicus.eu/how-to-api .

#### PyPI

It is easiest to install (LS)<sup>2</sup>D from PyPI:

    pip install ls2d
    
By default, this excludes the `cdsapi` as a dependency. If you do want to install that as a dependency, use:
    
    pip install ls2d[cds]
   
#### Manual

For a manual installation, you can clone the package from Github:

    git clone https://github.com/LS2D/LS2D.git

In each script where you want to use (LS)<sup>2</sup>D, add the (LS)<sup>2</sup>D root directory to the Python path:

    import sys
    sys.path.append('/path/to/LS2D')
    
You will have to manually install the dependencies with `pip install numpy scipy netCDF4 matplotlib cdsapi`.
    
### Usage

Some examples are provided at https://github.com/LS2D/LS2D/tree/main/examples. The script `example_1.py` downloads the ERA5 data, calculates the initial conditions and large scale forcings, and creates an example plot.

The examples directory also contains example cases for MicroHH (https://github.com/microhh/microhh).

### Downloading ERA5 data

(LS)<sup>2</sup>D contains two methods to download ERA5: through the [Copernicus Data Store](https://cds.climate.copernicus.eu) (open for everyone), or using MARS at ECMWF systems. 

The ERA5 model level data that (LS)<sup>2</sup>D requires is stored on tape archives, so downloads using CDS tend to be slow with long queueing times. For that reason, `ls2d.download_era5()` will stop the Python script once the download requests are submitted to CDS. On subsequent calls of `ls2d.download_era5()`, (LS)<sup>2</sup>D will check the status of the CDS request, and if the request is finished, download the ERA5 data. 

### The `settings` dictionary

All settings for (LS)<sup>2</sup>D are wrapped in a dictionary:

- `central_lat`: central latitude of LES/SCM domain
- `central_lon`: central longitude of LES/SCM domain
- `area_size`: spatial size of ERA5 download (central lat/lon +/- `area_size` degrees)
- `era5_path`: storage location of ERA5 downloads/data
- `era5_expver`: ERA5 experiment version number (`1`=normal ERA5, `5`=near realtime). With CDS, only `1` works.
- `case_name`: experiment name, only used to create subdirectory in `era5_path`.
- `start_date`: Python `datetime` object with start date/time
- `end_date`: Python `datetime` object with end date/time
- `write_log`: Write ERA5 download to screen (`False`) or log file (`True`)
- `data_source`: Download method (`CDS`, `MARS`, or `ARCO`). `MARS` only works on e.g. the ECMWF supercomputer. `ARCO` uses the [Google ARCO-ERA5](https://github.com/google-research/arco-era5) archive.

### Processing pipeline

The processing consists of stateless functions, which each take and return an `xarray.Dataset`:

```python
outputs = None                                                  # or e.g. ['thl', 'qt', 'ug', 'vg']
ls2d.download_era5(settings, outputs=outputs)                   # downloads only the ERA5 fields needed for `outputs`
raw = ls2d.read_era5(settings, outputs=outputs)                 # raw ERA5 fields (`ml:t`, `sfc:sp`, ...)
column = ls2d.calculate_forcings(raw, n_av=1, method='2nd', outputs=outputs)  # area averaged column + forcings
les_input = ls2d.get_les_input(column, z, outputs=outputs)      # interpolated to LES grid `z`
```

`outputs` selects the LES variables (default: all that the source can provide, `ls2d.era5.default_outputs()`). Pass the same list to every step. `ls2d.download()` and `ls2d.read()` do the same for any data source, selected with `settings['source']` (default `'era5'`). The old interface (`era = ls2d.Read_era5(settings)`, `era.calculate_forcings()`, `era.get_les_input(z)`) still works, and is a thin wrapper around these functions.

### GFS forecasts

Besides ERA5, (LS)<sup>2</sup>D can use NOAA GFS 0.25° forecasts, from the public archive on AWS (or its Google Cloud mirror); no account is needed. Install with `pip install ls2d[gfs]` (GRIB decoding with `eccodes`). The processing and the LES input are the same as for ERA5:

```python
settings = {
    'source': 'gfs',
    'central_lat': 51.97, 'central_lon': 4.93, 'area_size': 1, 'case_name': 'cabauw',
    'gfs_path': '/path/to/gfs/data',
    'gfs_cycle': datetime(2026, 10, 8, 0),    # optional, default: last cycle at or before `start_date`
    'start_date': datetime(2026, 10, 8, 6),   # valid times; hourly output up to +120 h, 3-hourly up to +384 h
    'end_date': datetime(2026, 10, 9, 6),
}
ls2d.download(settings)
column = ls2d.calculate_forcings(ls2d.read(settings), n_av=1)
les_input = ls2d.get_les_input(column, z)
```

Only the GRIB messages of the required fields are downloaded (HTTP byte ranges, ~300 MB and ~10 s per forecast hour), and saved cropped to the domain as one small NetCDF file per forecast hour. Differences with ERA5 to be aware of:

- The GFS atmosphere is only available on 41 pressure levels; it is interpolated to terrain-following levels (`p = sigma * ps`), ignoring levels below the surface, with the 2 m temperature/humidity and 10 m wind at the surface. The vertical resolution is that of the pressure levels (25 hPa near the surface).
- Surface fluxes in the GFS files are averages since the start of a 6 h window; (LS)<sup>2</sup>D converts them to the mean over the last output interval, valid at the end of the interval (ERA5: instantaneous). The analysis (+0 h) has no fluxes; the first interval mean is used.
- Land surface: Noah soil layers (0-0.1, 0.1-0.4, 0.4-1, 1-2 m) for `t_soil` and `theta_soil`; no HTESSEL vegetation/soil types, root fractions, vegetation cover, LAI, or `z0h`. SST is the surface temperature over water (undefined over land, as in ERA5).

See `examples/example_gfs.py`.

### Adding variables

Everything that (LS)<sup>2</sup>D downloads and computes is defined in registries (see `src/ls2d/forcing/`). Each quantity declares what it requires, and the registry works out what to download and in which order to compute things. Adding e.g. the 2 m temperature from ERA5 to the download, processing, and LES input:

```python
# 1. Raw ERA5 field, with its names for MARS / CDS / Google ARCO.
ls2d.era5.field('t2m', 'sfc', '167.128', cds='2m_temperature', arco='2m_temperature', units='K')

# 2. Quantity computed from it, here simply the field itself, averaged over the `n_av` area.
@ls2d.era5.quantity('t2m', requires=('sfc:t2m',), units='K', long_name='2 m temperature', reduce='mean')
def t2m(t2m, ctx):
    return t2m

# 3. Add to the LES input.
ls2d.les_output('t2m', 't2m', '2 m temperature', 'K')
```

Existing ERA5 files without the new field are downloaded again by `ls2d.download_era5()`. Quantities are either computed pointwise on the grid (`stage='field'`, the default), or from the averaging area including a halo for horizontal gradients (`stage='column'`). Quantities that only use standard quantities (see below) are source-agnostic, and are registered in the core with `ls2d.quantity`:

```python
@ls2d.quantity('dtT_advec', requires=('T', 'u', 'v'), stage='column', units='K s-1')
def dtT_advec(T, u, v, ctx):
    return ctx.advec(T, u, v)   # also available: ctx.mean(), ctx.ddx(), ctx.ddy(), ctx.nearest(), ctx.fc
```

### Data sources

(LS)<sup>2</sup>D separates the data source from the processing. A source (`ls2d.Source`, e.g. `ls2d.era5`) has its own raw fields, functions to download and read them, and recipes that turn them into a fixed set of *standard quantities* (`src/ls2d/forcing/standard.py`): `T`, `qv`, `ql`, `u`, `v`, `omega`, `p` or `ph`, `phi_p`, `ps`, `ts`, `wth`, `wq`, and optionally ozone, roughness lengths, soil, and land surface fields. Everything else (`thl`, `qt`, heights, advective tendencies, geostrophic wind, the LES input, ...) is derived by the source-agnostic core, so all sources produce the same LES input definition. The standard fixes units, dims (levels from surface to top, latitude south to north), and conventions (fluxes positive upward), and is checked when quantities are registered and computed.

To add a source (e.g. GFS), create `src/ls2d/sources/<name>.py` with:

```python
src = ls2d.register_source(ls2d.Source('gfs', read=read_gfs, download=download_gfs))
src.field('t', 'pl', units='K')                          # raw fields: `{levtype}:{name}`, levtype in ml/pl/sfc

@src.quantity('T', requires=('pl:t', 'ps'))               # recipes for the standard quantities
def T(t, ps, ctx):
    return to_terrain_following(t, ps, sigma)            # pressure levels -> terrain-following, see `ls2d.forcing.vertical`
...
src.validate()                                           # checks that all required standard quantities can be provided
```

`read_gfs(settings, outputs=None, fields=None)` returns the raw dataset (see `src/ls2d/forcing/raw.py`) with attribute `ls2d_source='gfs'`; `ls2d.read(dict(settings, source='gfs'))` then dispatches to it. `tests/synthetic_plev.py` is a complete example of a pressure level only source, which reproduces the ERA5 LES input within 2%. LES outputs that a source can not provide (e.g. the HTESSEL land surface types without HTESSEL fields) are left out automatically; land surface scheme specific outputs carry the attribute `land_surface_scheme`.

See `src/ls2d/sources/era5.py`, `src/ls2d/forcing/derived.py`, and `src/ls2d/forcing/column.py` for all built-in quantities.

## Contributing guidelines

Contributions through pull requests are always appreciated. To keep the code style consistent, we stick to PEP8 with two modifications (see [`project.toml`](./pyproject.toml)):

- The maximum line length is set to 120 characters instead of 80.
- Single quotes are used for strings instead of double quotes.

To ensure consistent formatting, the code should be automatically formatted using [ruff](https://docs.astral.sh/ruff/). It can be installed via pip and can be run on the entire codebase from the project root (this directory) like

    ruff format .

Tests (no ERA5 downloads needed, they use synthetic data) can be run with:

    pip install pytest
    pytest tests
