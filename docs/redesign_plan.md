# LS2D redesign plan: stateless, registry-driven ERA5 → LES pipeline

Status: **implemented** (steps 1–7), see *Implementation notes* at the end for where the
implementation differs from this plan. Step 8 (CAMS) is not done yet.
Base: `LS2D/LS2D@develop_arco` (`1941dd7`).

Decisions: base on `develop_arco`; keep the `Read_era5` wrapper; MARS requests (and all other
sources) only request the ERA5 fields needed for the requested outputs.

---

## 1. What exists today (and why it is hard to extend)

`Read_era5` (`src/ls2d/ecmwf/read_era5.py`) is one stateful object whose methods must be called in a fixed order,
each one adding attributes to `self` that the next one reads:

```
Read_era5(settings)                 # open files → ~45 self.<raw var>  → self.<derived var>
  .calculate_forcings(n_av, method) # → ~40 self.<var>_mean / self.<var>_nn, self.ug_mean, ...
  .get_les_input(z)                 # reads self.<var>_mean via getattr; fails if step 2 was skipped
```

Problems:

| Problem | Where |
|---|---|
| Hidden ordering: `get_les_input` silently depends on `calculate_forcings` having run (via `hasattr`/`getattr` on `<var>_mean`). | `read_era5.py:780-787` |
| One ERA5 variable is named in **5 places**: CDS request (long name), MARS request (param id), CDS model-level param id string, `read_data` (short name + flip/cast), plus the `var_4d_mean`/`var_3d_mean` lists and the `get_les_input` output table. Adding e.g. 2 m temperature means editing all of them. | `download_era5.py:255-301, 345-378`, `read_era5.py:228-272, 381-425, 758-775` |
| Inputs mutated: `settings['era5_path'] += '/'`. | `read_era5.py:92`, `download_era5.py:422` |
| Advection code duplicated 4× for the 4th-order scheme, geostrophic wind duplicated per method. | `read_era5.py:521-620` |
| Triple Python loops for half-level pressure/height and for `ug`/`vg` interpolation. | `read_era5.py:294-298, 508-516, 627-633` |
| No tests, so any refactor is unverified. | — |

## 2. Target architecture

Four explicit, pure stages. Each takes immutable inputs and returns a new `xarray.Dataset`; nothing is stored on an object.

```
            ┌──────────────── registry (declarative) ─────────────────┐
            │  ERA5 field catalogue      derived/forcing quantities   │
            │  sfc:sp, ml:t, pl:z, ...   thl, qt, p, ug, dtthl_advec  │
            └──────────────┬──────────────────────────┬───────────────┘
   requested outputs ──► resolve() ──► set of ERA5 fields
                                         │
 1. download_era5(settings, fields)      │  CDS / MARS / ARCO requests built from the catalogue
 2. read_era5(settings, fields)   ──► raw: xr.Dataset   (native grid, surface→top, S→N, SI units)
 3. compute(raw, outputs, ctx)    ──► column: xr.Dataset (time, level) + 3D fields if asked
      = derive_fields (pointwise, 3D)  +  calculate_forcings (3D sub-domain → column)
 4. get_les_input(column, z)      ──► les: xr.Dataset   (interpolated to LES grid, same names/attrs as today)
```

### 2.1 ERA5 field catalogue (download/read side)

One entry per raw ERA5 variable; the single source of truth for every backend.

```python
# src/ls2d/ecmwf/era5_fields.py
@dataclass(frozen=True)
class Era5Field:
    name: str          # short name in NetCDF, e.g. 't', 'sp'
    levtype: str       # 'ml' | 'pl' | 'sfc'  → decides which file / request it goes into
    cds: str           # CDS name ('surface_pressure') or ML param id ('130')
    mars: str          # MARS param ('134.128')
    arco: str | None   # Google ARCO name, if available
    units: str
    long_name: str
    read: Callable | None = None   # optional post-read transform, e.g. np.exp for flsr, round→int for slt

register_field(Era5Field('sp', 'sfc', 'surface_pressure', '134.128', 'surface_pressure', 'Pa', 'surface pressure'))
register_field(Era5Field('t',  'ml',  '130', '130.128', 'temperature', 'K', 'temperature'))
register_field(Era5Field('z',  'pl',  'geopotential', '129.128', 'geopotential', 'm2 s-2', 'geopotential'))
...
```

Keys are namespaced by level type (`'ml:t'`, `'pl:z'`, `'sfc:sp'`) because ERA5 short names collide
(`z` = geopotential on pressure levels vs. derived height `z`; `t` vs. `T`).

The CDS/MARS request builders become:
`variables = [f.cds for f in fields if f.levtype == 'sfc']` instead of hard-coded lists.
The reader becomes a loop over the same list. **Adding an ERA5 variable to download + read = one `register_field` line.**

### 2.2 Quantity registry (processing side)

A derived quantity declares what it needs and a pure function that computes it:

```python
# src/ls2d/forcing/registry.py
@dataclass(frozen=True)
class Quantity:
    name: str
    requires: tuple[str, ...]   # other quantities or ERA5 fields ('ml:t', 'sfc:sp', 'p', 'Tv', ...)
    func: Callable[..., xr.DataArray]   # func(*required, ctx) -> DataArray; must be pure
    stage: str                  # 'field'  : pointwise on the 3D grid
                                # 'column' : reduces the averaging sub-domain to a column (time, level)
    units: str
    long_name: str

def quantity(name, requires, stage='field', units='', long_name=''):  # decorator → register(Quantity(...))
```

Examples (moved 1:1 from current code):

```python
# forcing/thermo.py  (stage='field')
@quantity('ql', requires=('ml:clwc', 'ml:ciwc', 'ml:crwc', 'ml:cswc'), units='kg kg-1')
def ql(clwc, ciwc, crwc, cswc, ctx): return clwc + ciwc + crwc + cswc

@quantity('ph', requires=('sfc:sp',), units='Pa')            # vectorised: a + b * ps, no loops
@quantity('thl', requires=('ml:t', 'exn', 'ql'), units='K')
@quantity('wls', requires=('ml:w', 'rho'), units='m s-1')

# forcing/column.py  (stage='column', receives the sub-domain incl. a halo for gradients)
@quantity('dtthl_advec', requires=('thl', 'ml:u', 'ml:v'), stage='column', units='K s-1')
def dtthl_advec(thl, u, v, ctx): return advection(thl, u, v, ctx)     # one helper, 2nd or 4th order via ctx.method

@quantity('ug', requires=('pl:z', 'p'), stage='column', units='m s-1')
@quantity('root_frac_low_veg', requires=('sfc:tvl', 'sfc:slt'), stage='column')
```

Column quantities without a custom function (plain area mean or nearest neighbour) are generated from a
short table, so `thl_mean`, `ps_mean`, `soil_type_nn`, … do not need boilerplate.

`resolve(outputs)` walks the `requires` graph (topological sort, cycle detection, clear error for unknown names) and returns
(a) the evaluation order and (b) the leaf set of ERA5 fields → that set is what `download_era5` / `read_era5` fetch.
**Adding a forcing = one decorated function; its ERA5 inputs are then downloaded automatically.**

### 2.3 Context instead of `self`

Everything that used to come from `self`/`settings` during computation goes into an immutable value object:

```python
@dataclass(frozen=True)
class ForcingContext:
    central_lat: float
    central_lon: float
    n_av: int = 0
    method: str = '4th'          # '2nd' | '4th'
    ifs: IFS_tools = IFS_tools('L137')
    # derived lazily from (raw grid, lat/lon, n_av): i, j, sub-domain slices, dx, dy, fc
```

### 2.4 LES output table

`get_les_input(column, z)` becomes a loop over a declarative table, so the output contract is visible in one place
and kept **identical** to today's dataset (names, dims, `long_name`, `units`, global attrs):

```python
LES_OUTPUTS = [
    LesVar('thl',         src='thl',          how='interp_z'),
    LesVar('dtthl_advec', src='dtthl_advec',  how='interp_z'),
    LesVar('p_lay',       src='p',            how='lay'),
    LesVar('p_lev',       src='ph',           how='lev'),
    LesVar('type_soil',   src='type_soil',    how='scalar'),
    LesVar('t_soil',      src='T_soil',       how='zs'),
    ...
]
```

`get_les_input` checks that `column` contains every `src` it needs and raises a clear error otherwise — the
"call `calculate_forcings` first" requirement becomes an explicit data dependency rather than hidden state.

### 2.5 Public API

```python
outputs = ls2d.default_les_outputs()                 # or a user list, e.g. + ['t2m']
fields  = ls2d.required_era5_fields(outputs)
ls2d.download_era5(settings, fields=fields)          # fields defaults to default outputs → no change for users
raw     = ls2d.read_era5(settings, fields=fields)
column  = ls2d.calculate_forcings(raw, ctx)          # ctx = ForcingContext(lat, lon, n_av=1, method='2nd')
les     = ls2d.get_les_input(column, z)
```

Backwards compatibility: `Read_era5` stays as a thin wrapper (`__init__` → `read_era5`,
`calculate_forcings` → stores the returned column, `get_les_input` → calls the pure function).
Existing scripts in `examples/` keep working unchanged; the old attributes (`era.thl`, `era.z_mean`, …) are
exposed via `__getattr__` onto the datasets during a deprecation period.

### 2.6 Proposed layout

```
src/ls2d/
  forcing/
    registry.py      Quantity, quantity(), resolve(), evaluate()
    context.py       ForcingContext, sub-domain/halo selection, dx/dy
    thermo.py        field-stage quantities (ql, qt, Tv, ph, zh, p, z, exn, th, thl, rho, wls, wth, h2o, o3 vmr, soil stacks)
    column.py        column-stage quantities (means, nn, advection, geostrophic wind, Th, root fractions)
    les.py           LES_OUTPUTS + get_les_input()
  ecmwf/
    era5_fields.py   ERA5 field catalogue
    download_era5.py request builders read the catalogue (CDS / MARS)
    read_era5.py     read_era5() → raw Dataset; Read_era5 compat wrapper
tests/
  conftest.py        synthetic ERA5 files (small domain, 3 times, L137) in the real on-disk layout
  test_regression.py new pipeline == current Read_era5 output (golden file generated from main before refactor)
  test_registry.py   resolve(), cycles, unknown names, required-field sets
  test_requests.py   generated CDS/MARS request dicts == current hard-coded ones
```

## 3. Implementation steps

Each step keeps the examples working and is checked by the regression test.

1. **Safety net.** Add `tests/` with a synthetic ERA5 dataset written in the real file layout; run the *current*
   `Read_era5 → calculate_forcings(2nd and 4th) → get_les_input` on it and store the result as golden NetCDF.
   Add a test that snapshots the CDS/MARS request dicts.
2. **Registry core** (`registry.py`, `context.py`) with unit tests; no behaviour change.
3. **ERA5 catalogue**; rewrite request builders and the reader on top of it → `read_era5()` returns the raw Dataset.
   Request-snapshot test must stay green.
4. **Field-stage quantities** (`thermo.py`), vectorising the `ph`/`zh` loops.
5. **Column-stage quantities** (`column.py`): one `advection()` helper for both orders, one geostrophic-wind helper,
   vectorised `ug`/`vg` interpolation. Public `calculate_forcings(raw, ctx)`.
6. **`get_les_input(column, z)`** driven by `LES_OUTPUTS`.
7. **Compat wrapper** `Read_era5`, update examples/README to show both styles, add a "how to add a variable" section.
8. (Later, separate PR) Move `Read_cams` onto the same registry; share the catalogue with the ARCO downloader.

Acceptance: regression test equal to golden within `rtol=1e-10` for both methods (only float-order differences from
vectorisation), request snapshots unchanged, examples import and run against the synthetic data.

## 4. Example: adding a new variable after the redesign

```python
# 1. raw ERA5 field (download + read for CDS/MARS/ARCO)
register_field(Era5Field('t2m', 'sfc', '2m_temperature', '167.128', '2m_temperature', 'K', '2 m temperature'))

# 2. forcing quantity (column mean of the field)
@quantity('t2m', requires=('sfc:t2m',), stage='column', units='K', long_name='2 m temperature')
def t2m(t2m, ctx): return ctx.mean(t2m)

# 3. optional: expose to LES
LES_OUTPUTS.append(LesVar('t2m', src='t2m', how='time'))
```

## 5. Choosing the base (decision needed)

`LS2D/LS2D` has a newer branch, **`develop_arco`** (4 commits ahead of `main`, 0 behind, last commit 2026-09-24,
published as a PyPI pre-release). It already:

- adds a Google ARCO-ERA5 downloader/reader,
- introduces a "generic 3D dataset" spec (`ls2d/column/spec.py`) and a stateless
  `create_column_input(ds, z, n_av)` (2nd-order only) — i.e. step 2/3 of the pipeline above, without a registry,
- replaces `core.messages` with `logging`.

The plan above fits on either base. On `develop_arco`, `spec.py`'s `required`/`optional` tables would become the
registry's quantity metadata, `create_column_input` would split into `calculate_forcings` + `get_les_input`,
and the ARCO downloader's hard-coded `_vars_ml/_vars_pl/_vars_sfc` dicts would read the `arco` column of the catalogue.
Building on `main` instead risks a large conflict with that branch when it lands.

## 6. Open questions

1. Base branch: `main` or `develop_arco`? (Recommendation: `develop_arco`.)
2. Keep the `Read_era5` compatibility wrapper (recommended), or make a clean break?
3. MARS surface requests currently fetch ~55 parameters, CDS only the 22 used. With the catalogue both would request
   exactly what the requested outputs need (plus an `extra_fields` option). OK to shrink the MARS request?
4. Keep NetCDF-file-per-levtype layout on disk (`model_an.nc`, `pressure_an.nc`, `surface_an.nc`) — recommended, so
   existing downloads stay valid. Re-downloading is only needed when a new field is added; the reader would then
   report which fields are missing from existing files.

---

## 7. Implementation notes

Layout as implemented (after the source abstraction, section 8):

```
src/ls2d/forcing/
  registry.py     Field, Quantity, Registry (parent fallback, resolve / evaluate), `core` registry
  standard.py     the standard quantities: contract between sources and core
  constants.py    physical constants of the core
  derived.py      source-agnostic field quantities (Tv, thl, qt, z, zh, w, h2o, ..., p <-> ph fallbacks)
  column.py       column quantities (advection, geostrophic wind, Coriolis, Th)
  vertical.py     vertical helpers (pressure levels -> terrain-following, half levels)
  les.py          LES output table, get_les_input()
  context.py      Context (settings + operators) and Domain (averaging area + halo)
  source.py       Source, register_source(), get_source()
  raw.py          raw dataset conventions, shared by the readers
  pipeline.py     required_fields(), download(), read(), compute_fields(), calculate_forcings()
src/ls2d/sources/
  era5.py         ERA5 source: Era5Field, catalogue (CDS/MARS/ARCO names), ERA5 recipes
```

Differences from the plan above:

- **ERA5 keys** are `{levtype}:{name}` (`ml:t`, `pl:z`, `sfc:sp`); the raw dataset uses the same keys as variable names.
- **Column variables keep the field names** (`thl`, not `thl_mean`). A field quantity's `reduce` attribute
  (`'mean'`, `'nearest'` or a function) decides how it appears in the column dataset; there is no separate table.
- **Interpolated LES outputs also require the column `z`**, so e.g. `outputs=['ug']` still downloads the
  model-level thermodynamics (`ls2d.registry.required_era5_fields(['ug'])` = `pl:z`, `sfc:sp` only).
- **Physics follows the `develop_arco` ARCO path** where it differed from the legacy `Read_era5`:
  `h2o_lay` uses `qv` instead of `qt`, and `wq` is divided by the surface density (legacy returned kg m-2 s-1).
  The long names of the released version are kept (`ECMWF soil type ...`).
- **Small domains:** with `method='2nd'` the halo shrinks to what is available (one-sided gradients at the edge,
  as in the legacy code, with a warning). `method='4th'` needs the full halo.
- **Defaults:** `calculate_forcings(method='2nd')` (as `create_column_input`), `Read_era5.calculate_forcings(method='4th')`
  (as before).
- **Downloads:** files that exist but miss a required field are downloaded again, including the fields already in them.
  `download_era5(settings)` also dispatches to ARCO with `data_source='ARCO'`.
- **`develop_arco` API:** `read_era5_arco()` returns `compute_fields(raw)` (a superset of the old generic dataset), and
  `create_column_input()` is a shortcut for `get_les_input(calculate_forcings(...))`. `ls2d.column.validate` now checks
  the raw dataset.
- `Read_era5` keeps its three methods; the old per-variable attributes (`era.thl`, `era.z_mean`, ...) are replaced by
  `era.raw` and `era.column` (none of the examples used them).

Verification (`pytest tests`, synthetic ERA5 in the real file layouts, no downloads needed): the new pipeline and the
`Read_era5` wrapper reproduce the pre-refactor output (`tests/data/golden_*.nc`, from `tests/make_golden.py`) to
1e-10 for 2nd and 4th order and for a 3×3 domain, except the two deliberate physics changes; the ARCO path matches to
float32 precision. Requests are checked against the old hard-coded CDS lists.

## 8. Source abstraction (follow-up)

Goal: adding e.g. GFS gives the same LES input definition, without touching the core.

- **`Source`** (`forcing/source.py`): name, raw field class, `download()`/`read()`, and a registry whose
  parent is the `core` registry. A source registers its raw fields and the recipes for the standard quantities;
  names it does not define come from the core, and it can override core quantities. `ls2d.read()` /
  `ls2d.download()` dispatch on `settings['source']`; `calculate_forcings()` on the raw dataset's `ls2d_source`.
- **Standard** (`forcing/standard.py`): sources provide `T`, `qv`, `ql`, `u`, `v`, `omega`, `p` or `ph`, `phi_p`,
  `ps`, `ts`, `wth`, `wq` (+ optional ozone, roughness, soil, land surface). The core derives `Tv`, `zh`, `z`, `exn`,
  `thl`, `qt`, `rho`, `w`, `h2o`, `rhos`. Registering a standard quantity with other units is an error; computed
  values must have the standard dims (checked in `Registry.evaluate`). `Source.validate()` lists what is missing.
- **Core fallbacks**: `p` from `ph` and `ph` from (`p`, `ps`); `ls2d.forcing.vertical.to_terrain_following()` converts
  pressure level data to `p = sigma * ps` levels, ignoring levels below the surface.
- **Scheme-specific outputs**: HTESSEL types and root fractions are tagged (`land_surface_scheme='HTESSEL'`); a source
  without them simply does not provide those outputs (`Source.default_outputs()`).
- **Proof** (`tests/test_sources.py`): `tests/synthetic_plev.py` is a GFS-like source (pressure levels only, one
  condensate species, upward fluxes in W m-2, no hybrid levels, no HTESSEL). It gives the same variables, dims, units,
  and attributes as ERA5, and values within 2% of the field maximum (37 pressure levels vs. 137 model levels).
- The ERA5 constants moved from `IFS_tools` to `forcing/constants.py` (same values); ERA5 output is unchanged (1e-10).
- While adding this, a bug in the synthetic test data was found (geopotential gradient constant with height, so
  `ug`/`vg` were not really tested); fixed, and the golden files regenerated with the pre-refactor code.

## 9. GFS source

`ls2d.sources.gfs` + `ls2d.noaa` (download/read): NOAA GFS 0.25 degree forecasts from the public archive
on AWS (`noaa-gfs-bdp-pds`) or its Google Cloud mirror.

- Download: `.idx` inventories -> only the GRIB messages of the required fields (HTTP byte ranges, merged,
  parallel download + eccodes decoding, ~10 s per forecast hour), cropped to the domain, one NetCDF per hour
  (`gfs_path/case/gfs/yyyymmddhh/gfs.fXXX.nc`). Files missing fields are downloaded again (keeping existing fields).
- Recipes: pressure levels -> terrain-following levels (2 m / 10 m values at the surface), condensate (5 species,
  zero above 50 hPa), upward fluxes, Noah soil, SST from skin temperature over water.
- Fluxes are window averages in GFS; the reader converts them to interval means (the hour before the first
  output time is downloaded for that, fluxes only).
- Tests: offline with a synthetic GRIB2 archive (`tests/synthetic_gfs.py`, real GRIB messages via eccodes, fake
  `fetch`); live test with `LS2D_LIVE=1`. Checked manually against real GFS data, and against ERA5 (ARCO).
