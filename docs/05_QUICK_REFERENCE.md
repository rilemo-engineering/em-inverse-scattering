# Quick Reference Card: Inverse Scattering Exercises

A concise summary of key formulas, parameters, and guidelines.

---

## Physical Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $\varepsilon_0$ | $8.85 \times 10^{-12}$ F/m | Vacuum permittivity |
| $\mu_0$ | $4\pi \times 10^{-7}$ H/m | Vacuum permeability |
| $c$ | $3 \times 10^8$ m/s | Speed of light |

---

## Key Formulas

### Wavelength

$$\lambda_0 = \frac{c}{f}$$

Example: $f = 4$ GHz $\Rightarrow \lambda_0 = 0.075$ m $= 7.5$ cm

### Wavenumber

$$k_0 = \frac{2\pi}{\lambda_0} = \omega\sqrt{\varepsilon_0\mu_0}$$

$$k_b = k_0\sqrt{\varepsilon_b} \quad \text{(in background medium)}$$

### Contrast Function

$$\tau(\mathbf{r}) = \varepsilon_r(\mathbf{r}) - \varepsilon_b - j\frac{\sigma(\mathbf{r})}{\omega\varepsilon_0}$$

- $\varepsilon_r = 3$, $\varepsilon_b = 1$, $\sigma = 0$ $\Rightarrow$ $\tau = 2$
- $\varepsilon_r = 1.5$, $\varepsilon_b = 1$, $\sigma = 0$ $\Rightarrow$ $\tau = 0.5$

### Degrees of Freedom

$$\text{DoF} \approx 2 k_0 a = \frac{4\pi a}{\lambda_0}$$

where $a$ = characteristic object size (e.g., $a = \frac{\sqrt{2} \cdot l_x}{2}$ for square DoI)

### Resolution Limit

$$\delta \approx \frac{\lambda_0}{2} \quad \text{(with full 360° coverage)}$$

### TSVD Solution

$$\boldsymbol{\tau}_{\text{TSVD}} = \sum_{i=1}^{k} \frac{\mathbf{u}_i^T \mathbf{E}_{\text{scat}}}{\sigma_i} \mathbf{v}_i$$

### Truncation Threshold ↔ Index

$$\text{Threshold [dB]} = 20 \cdot \log_{10}\left(\frac{\sigma_k}{\sigma_1}\right)$$

### NMSE

$$\text{NMSE} = \frac{\|\tau_{\text{true}} - \tau_{\text{rec}}\|^2}{\|\tau_{\text{true}}\|^2}$$

---

## Parameter Quick Guide

### Simulated Data (Typical)

| Parameter | Symbol | Typical Values |
|-----------|--------|----------------|
| Frequency | $f$ | 300 MHz - 10 GHz |
| DoI side | $l_x$, $l_y$ | 0.1 - 1.0 m |
| Grid points | $N_x$, $N_y$ | 32 - 128 |
| Measurement radius | $R_m$ | $> l_x \cdot \sqrt{2}/2$ |
| Receivers | $N_m$ | $\approx$ DoF |
| Transmitters | $N_v$ | $\approx N_m$ |
| Background $\varepsilon$ | $\varepsilon_b$ | 1 (free space) |
| Background $\sigma$ | $\sigma_b$ | 0 S/m |

### Fresnel Experimental Data

| Parameter | Value |
|-----------|-------|
| $f$ | 4 GHz |
| $\lambda_0$ | 7.5 cm |
| $l_x$, $l_y$ | 15 cm |
| $N_x$, $N_y$ | 64 |
| $R_m$ | 76.1 cm |
| $R_v$ | 72.1 cm |
| $N_m$ | 49 |
| $N_v$ | 36 |

### Single Cylinder Target
- Radius: 15 mm
- Position: (25 mm, 0)
- $\varepsilon_r$: $3.0 \pm 0.3$
- $\tau$: $2.0 \pm 0.3$

### Two Cylinder Target
- Both radius: 15 mm
- Left: $(-45$ mm$, 15$ mm$)$
- Right: $(45$ mm$, 5$ mm$)$
- Both $\varepsilon_r$: $3.0 \pm 0.3$

---

## Truncation Threshold Guidelines

### Starting Point Rule

$$\text{Threshold [dB]} \approx -(\text{SNR} - 5)$$

| SNR [dB] | Start Threshold [dB] |
|----------|---------------------|
| 40 | $-35$ |
| 30 | $-25$ |
| 20 | $-15$ |
| 10 | $-5$ |

### Effect of Threshold

| Threshold | Effect |
|-----------|--------|
| Too high (e.g., $-10$ dB) | Blurry, over-smoothed |
| Optimal | Balance resolution/noise |
| Too low (e.g., $-50$ dB) | Noisy, artifacts |

---

## NMSE Interpretation

| NMSE | Quality |
|------|---------|
| $< 0.05$ | Excellent |
| $0.05$ - $0.15$ | Good |
| $0.15$ - $0.30$ | Acceptable |
| $0.30$ - $0.50$ | Marginal |
| $> 0.50$ | Poor |

**Note:** For experimental data, expect NMSE 10-50% higher than simulated.

---

## MATLAB Variable Reference

### Input Variables (from forward solver)

| Variable | Size | Description |
|----------|------|-------------|
| `Escat` | $N_m \times N_v$ | Scattered field data |
| `Einc_domain` | $N_y \times N_x \times N_v$ | Incident field on DoI |
| `Etot_domain` | $N_y \times N_x \times N_v$ | Total field on DoI |
| `PROF` | $N_y \times N_x$ | True contrast profile |
| `freq` | $1 \times 1$ | Frequency [Hz] |
| `lx`, `ly` | $1 \times 1$ | DoI dimensions [m] |
| $N_x$, $N_y$ | $1 \times 1$ | Grid points |
| $\varepsilon_b$, $\sigma_b$ | $1 \times 1$ | Background properties |
| $R_m$ | $1 \times 1$ | Measurement radius [m] |

### Inversion Variables

| Variable | Size | Description |
|----------|------|-------------|
| `S_BORN` | $(N_m \cdot N_v) \times (N_x \cdot N_y)$ | Scattering operator |
| $\mathbf{U}$, $\mathbf{V}$ | matrices | SVD components |
| $\boldsymbol{\Sigma}$ | diagonal | Singular values |
| $N_t$ | $1 \times 1$ | Truncation index |
| `PROF_rec_BORN` | $N_y \times N_x$ | Reconstructed profile |

---

## Workflow Checklists

### Forward Problem
- [ ] Set scenario parameters
- [ ] Run `c1_Scenario.m`
- [ ] Visualize profile and fields
- [ ] Check MVMS matrix structure
- [ ] Save `DATA_scenario.mat`

### Inverse Problem (Simulated)
- [ ] Load `DATA_scenario.mat`
- [ ] Set SNR level
- [ ] Set `AL = 0` or `1`
- [ ] Run `c2_Inversion_BORN.m`
- [ ] Examine singular value spectrum
- [ ] Choose threshold (start with rule of thumb)
- [ ] Evaluate NMSE
- [ ] Compare reconstruction to ground truth
- [ ] Iterate threshold if needed

### Inverse Problem (Experimental)
- [ ] Select dataset (`dielTM_dec8f.txt` or `twodielTM_8f.txt`)
- [ ] Run `c1_Scenario_ExpData.m`
- [ ] Load matching scenario files
- [ ] Run `c2_Inversion_ExpData_BORN.m`
- [ ] Choose threshold (may need trial)
- [ ] Check cross-sections against tolerances
- [ ] Visually validate position and shape

---

## Common Errors and Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| Reconstruction explodes | Threshold too low | Increase (less negative) |
| Reconstruction too blurry | Threshold too high | Decrease (more negative) |
| Wrong position | Data mismatch | Check coordinate systems |
| NMSE $> 1$ | Strong scatterer | Born approximation invalid |
| No convergence | Too few iterations | Increase `n_iter` |

---

## Born Approximation Validity

**Valid when:**
- Object size $< \lambda$
- $|\tau| < 1$ (weak contrast)
- Single scattering dominates

**Invalid when:**
- Object size $> \lambda$
- $|\tau| > 1$ (strong contrast)
- Multiple scattering significant

---

## Key Equations Summary

| Equation | Name |
|----------|------|
| $\nabla^2 E + k^2 E = 0$ | Helmholtz equation |
| $E_{\text{tot}} = E_{\text{inc}} + k_0^2 \iint \tau E_{\text{tot}} G \, d\mathbf{r}'$ | Lippmann-Schwinger |
| $G = \frac{i}{4} H_0^{(1)}(k_b \|\mathbf{r}-\mathbf{r}'\|)$ | 2D Green's function |
| $\tau = \varepsilon_r - \varepsilon_b - i\frac{\sigma}{\omega\varepsilon_0}$ | Contrast function |
| $E_{\text{tot}} \approx E_{\text{inc}}$ | Born approximation |
| $\mathbf{E}_{\text{scat}} = \mathbf{S} \boldsymbol{\tau}$ | Linearized inverse problem |
| $\mathbf{S} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | SVD decomposition |
| $\boldsymbol{\tau}_{\text{TSVD}} = \sum_{i=1}^{k} \frac{\mathbf{u}_i^T \mathbf{E}_{\text{scat}}}{\sigma_i} \mathbf{v}_i$ | TSVD solution |
| $\text{NMSE} = \frac{\|\tau_{\text{true}} - \tau_{\text{rec}}\|^2}{\|\tau_{\text{true}}\|^2}$ | Quality metric |

---

## File Quick Reference

```
matlab/simulated/
├── scenario/
│   └── c1_Scenario.m           → Generate synthetic data
├── inversion/
│   └── c2_Inversion_BORN.m     → Reconstruct from simulated

matlab/experimental/
├── scenario/
│   └── c1_Scenario_ExpData.m   → Load Fresnel data
├── inversion/
│   └── c2_Inversion_ExpData_BORN.m → Reconstruct from experimental
```

---

*Keep this reference handy while working through the exercises!*
