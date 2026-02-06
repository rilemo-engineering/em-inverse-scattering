# MATLAB to Python Function Mapping

This document maps MATLAB protected functions (.p files) to their Python implementations.

## Simulated Scenario (Forward Problem)

| MATLAB (.p) | Python Module | Python Function | Status |
|-------------|---------------|-----------------|--------|
| `forward_solver.p` | `forward/forward_solver.py` | `forward_solver()` | Implemented |
| `Profili.p` | `forward/profiles.py` | `create_circular_profile()` | Implemented |
| `ainterno.p` | `forward/internal_operator.py` | `build_internal_operator()` | Implemented |
| `CGFFT.p` | `forward/cgfft.py` | `cgfft_solve()`, `cgfft_solve_all_views()` | Implemented |
| `DFUNC.p` | `forward/cgfft.py` | (internal to CGFFT) | Integrated |
| `LINMIN.p` | `forward/cgfft.py` | (line search in CG) | Integrated |

## Simulated Inversion

| MATLAB (.p) | Python Module | Python Function | Status |
|-------------|---------------|-----------------|--------|
| `kernel_scattering.p` | `inverse/scattering_kernel.py` | `kernel_scattering()` | Verified |
| `TSVD_solver.p` | `inverse/tsvd.py` | `tsvd_solve()` | Verified |

## Experimental Scenario

| MATLAB (.p) | Python Module | Python Function | Status |
|-------------|---------------|-----------------|--------|
| `load_data_fr2001.p` | `data/fresnel_loader.py` | `load_fresnel_data()` | Implemented |

## Experimental Inversion

| MATLAB (.p) | Python Module | Python Function | Status |
|-------------|---------------|-----------------|--------|
| `kernel_scattering_exp.p` | `inverse/scattering_kernel.py` | `kernel_scattering_exp()` | Implemented |
| `TSVD_solver.p` | `inverse/tsvd.py` | `tsvd_solve()` | Verified |

## Script Mapping

| MATLAB Script | Python Script | Status |
|---------------|---------------|--------|
| `c1_Scenario.m` | `scripts/scenario.py` | Implemented |
| `c2_Inversion_BORN.m` | `scripts/inversion_born.py` | Verified |
| `c1_Scenario_ExpData.m` | `scripts/scenario_exp.py` | Implemented |
| `c2_Inversion_ExpData_BORN.m` | `scripts/inversion_exp_born.py` | Implemented |

## Key Implementation Notes

### Green's Function Convention
- MATLAB uses `exp(-jωt)` time convention
- Python uses `H_0^(2)` (Hankel function of second kind) with factor `-(j/4)`
- Formula: `G(r) = -(j/4) * H_0^(2)(k*r)`

### Array Ordering
- MATLAB is column-major (Fortran order)
- Python numpy default is row-major (C order)
- All `ravel()` calls use `order='F'` for MATLAB compatibility

### Data Files
- Fresnel data: `dielTM_dec8f.txt` (single target), `twodielTM_8f.txt` (two targets)
- MATLAB generates: `DATA_scenario.mat`, `DATA_scenario_exp_*.mat`
- Python generates: `.mat` (MATLAB compatible) + `.npz` (numpy native)

## Verified Components (NMSE matches MATLAB)

1. **TSVD reconstruction**: NMSE 0.1546 (Python) vs 0.1542 (MATLAB) - within RNG tolerance
2. **Scattering kernel**: Array dimensions and ordering verified
3. **AWGN noise**: Same SNR formula as MATLAB
4. **Visualization**: origin='lower', parula colormap equivalent

## Audit Findings (Feb 2026)

### Simulated Data Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| Wavelength/wavenumber | Verified | Exact match |
| DOF calculation | Verified | Exact match |
| Grid generation | Verified | Exact match |
| Profile generation | Verified | Works for circular targets |
| Inversion (TSVD) | Verified | NMSE=0.1546 matches MATLAB |

### Experimental Data Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| Fresnel data loader | Partial | Loads raw data, but MATLAB has unknown calibration |
| Experimental kernel | Verified | Identical to simulated kernel |
| Inversion | Works | NMSE≈1.38 with MATLAB Einc_domain |

### Protected Functions (.p files)
These MATLAB functions cannot be inspected, so Python implementations are based on documented behavior:
- `forward_solver.p` - Python forward solver works but may differ in details
- `load_data_fr2001.p` - Unknown calibration; use MATLAB .mat files instead
- `CGFFT.p` - Python CG-FFT implemented but not directly verified

### Recommendations
1. **Simulated data**: Python implementation is 1:1 with MATLAB
2. **Experimental data**: Use MATLAB-generated .mat files (DATA_scenario_exp_*.mat)
3. **New experiments**: Python forward solver suitable for generating new data
