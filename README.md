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

`outputs` selects the LES variables (`ls2d.les_outputs()` lists all of them, which is the default). Pass the same list to every step. The old interface (`era = ls2d.Read_era5(settings)`, `era.calculate_forcings()`, `era.get_les_input(z)`) still works, and is a thin wrapper around these functions.

### Adding variables

Everything that (LS)<sup>2</sup>D downloads and computes is defined in a registry (see `src/ls2d/forcing/`). Each quantity declares what it requires, and the registry works out what to download and in which order to compute things. Adding e.g. the 2 m temperature to the download, processing, and LES input:

```python
# 1. Raw ERA5 field, with its names for MARS / CDS / Google ARCO.
ls2d.era5_field('t2m', 'sfc', '167.128', cds='2m_temperature', arco='2m_temperature', units='K')

# 2. Quantity computed from it, here simply the field itself, averaged over the `n_av` area.
@ls2d.quantity('t2m', requires=('sfc:t2m',), units='K', long_name='2 m temperature', reduce='mean')
def t2m(t2m, ctx):
    return t2m

# 3. Add to the LES input.
ls2d.les_output('t2m', 't2m', '2 m temperature', 'K')
```

Existing ERA5 files without the new field are downloaded again by `ls2d.download_era5()`. Quantities are either computed pointwise on the ERA5 grid (`stage='field'`, the default), or from the averaging area including a halo for horizontal gradients (`stage='column'`), for example:

```python
@ls2d.quantity('dtT_advec', requires=('T', 'u', 'v'), stage='column', units='K s-1')
def dtT_advec(T, u, v, ctx):
    return ctx.advec(T, u, v)   # also available: ctx.mean(), ctx.ddx(), ctx.ddy(), ctx.nearest(), ctx.fc
```

See `src/ls2d/forcing/fields.py` and `src/ls2d/forcing/column.py` for all built-in quantities.

## Contributing guidelines

Contributions through pull requests are always appreciated. To keep the code style consistent, we stick to PEP8 with two modifications (see [`project.toml`](./pyproject.toml)):

- The maximum line length is set to 120 characters instead of 80.
- Single quotes are used for strings instead of double quotes.

To ensure consistent formatting, the code should be automatically formatted using [ruff](https://docs.astral.sh/ruff/). It can be installed via pip and can be run on the entire codebase from the project root (this directory) like

    ruff format .

Tests (no ERA5 downloads needed, they use synthetic data) can be run with:

    pip install pytest
    pytest tests
