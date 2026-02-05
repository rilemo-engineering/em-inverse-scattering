# Exercise Guide: Experimental Data (Fresnel Institute)

This guide walks you through the experimental data exercises using real measurements from the Fresnel Institute.

---

## Table of Contents

1. [About the Fresnel Dataset](#1-about-the-fresnel-dataset)
2. [Differences from Simulated Data](#2-differences-from-simulated-data)
3. [File Structure and Data Loading](#3-file-structure-and-data-loading)
4. [Running the Exercises](#4-running-the-exercises)
5. [Understanding the Targets](#5-understanding-the-targets)
6. [Exercises and Experiments](#6-exercises-and-experiments)
7. [Validation and Benchmarking](#7-validation-and-benchmarking)

---

## 1. About the Fresnel Dataset

### What is it?

The **Fresnel Institute** (Institut Fresnel, Marseille, France) has provided benchmark experimental datasets for the inverse scattering community since the early 2000s. These are high-quality, carefully calibrated measurements of simple objects.

### Why Use Experimental Data?

| Simulated Data | Experimental Data |
|----------------|-------------------|
| Perfect forward model | Real-world physics |
| Known ground truth exactly | Ground truth has tolerances |
| Controlled noise (AWGN) | Complex noise + systematics |
| Tests algorithm only | Tests algorithm + physics model |

**Key insight:** An algorithm that works perfectly on simulated data may struggle with real data due to:
- Model mismatch
- Calibration errors
- Non-Gaussian noise
- Systematic errors

### Dataset Provenance

The datasets used here are from:
> J.-M. Geffrin, P. Sabouroux, C. Eyraud, "Free space experimental scattering database continuation: experimental set-up and measurement precision," *Inverse Problems*, vol. 21, S117, 2005.

**Measurement setup:**
- Anechoic chamber at Centre Commun de Ressources Micro-ondes (CCRM)
- Frequencies: 2-10 GHz
- TM polarization
- Circular measurement configuration

---

## 2. Differences from Simulated Data

### What's Different

| Aspect | Simulated | Experimental |
|--------|-----------|--------------|
| $E_{\text{inc}}$ | Computed analytically | Measured/calibrated |
| $E_{\text{scat}}$ | From forward solver | Direct measurement |
| Noise | Added artificially (AWGN) | Inherent in measurement |
| Ground truth | Exact (you defined it) | Known with tolerances |
| Geometry | Exact coordinates | Slight misalignments possible |
| Object properties | Exact contrast | Measured $\varepsilon_r \pm$ uncertainty |

### Implications for Inversion

1. **No "true" SNR** - you don't know the exact noise level
2. **Model mismatch** - 2D assumption may not be perfect
3. **Calibration residuals** - multiplicative errors possible
4. **Contrast uncertainty** - target $\varepsilon_r = 3.0 \pm 0.3$ (nominal)

### What to Expect

- NMSE typically higher than for simulated data
- May need different truncation threshold
- Reconstruction quality still good if model is appropriate

---

## 3. File Structure and Data Loading

### Folder Structure

```
matlab/experimental/
├── scenario/
│   ├── c1_Scenario_ExpData.m      ← Load and visualize data
│   ├── load_data_fr2001.p         ← Data loader (protected)
│   ├── dielTM_dec8f.txt           ← Single cylinder data
│   ├── twodielTM_8f.txt           ← Two cylinders data
│   └── DATA_scenario_exp_*.mat    ← Processed output
│
├── inversion/
│   ├── c2_Inversion_ExpData_BORN.m  ← Inversion script
│   ├── kernel_scattering_exp.p      ← Scattering kernel
│   ├── TSVD_solver.p                ← Same as simulated
│   └── DATA_*.mat                   ← Scenario data
```

### Key Parameters (Set in Code)

```matlab
eb = 1;           % Background: free space
sb = 0;           % No background conductivity
freq = 4e9;       % 4 GHz (one of available frequencies)
lambda0 = 0.075;  % = c/freq = 7.5 cm
lx = 0.15;        % 15 cm DoI side
Nx = Ny = 64;     % 64×64 grid
Rv = 0.72135;     % Transmitter radius [m]
Rm = 0.76135;     % Receiver radius [m]
```

### Available Datasets

| File | Target | Description |
|------|--------|-------------|
| `dielTM_dec8f.txt` | Single cylinder | 15mm radius, $\varepsilon_r \approx 3$, offset from center |
| `twodielTM_8f.txt` | Two cylinders | Both 15mm radius, $\varepsilon_r \approx 3$, 90mm apart |

---

## 4. Running the Exercises

### Step 1: Load and Visualize Data

1. Navigate to `matlab/experimental/scenario/`

2. Open `c1_Scenario_ExpData.m` and select dataset:
   ```matlab
   dataset = 'dielTM_dec8f.txt';    % Single cylinder
   % dataset = 'twodielTM_8f.txt';  % Two cylinders
   ```

3. Run the script:
   ```matlab
   >> c1_Scenario_ExpData
   ```

4. Observe the plots showing:
   - Known target profile (from specifications)
   - Cross-sections through the target

### Step 2: Run Inversion

1. Navigate to `matlab/experimental/inversion/`

2. Open `c2_Inversion_ExpData_BORN.m`

3. Select matching scenario data:
   ```matlab
   load DATA_scenario_exp_singletarget.mat
   load DATA_object_exp_singletarget.mat
   % OR for two targets:
   % load DATA_scenario_exp_twotargets.mat
   % load DATA_object_exp_twotargets.mat
   ```

4. Run and choose truncation threshold when prompted

5. Examine reconstruction plots

---

## 5. Understanding the Targets

### Single Dielectric Cylinder

**Specifications:**
- Radius: $r_0 = 15$ mm $= 0.2\lambda$ at 4 GHz
- Center: $(x_0, y_0) = (25 \text{ mm}, 0 \text{ mm})$
- Material: $\varepsilon_r = 3.0 \pm 0.3$
- Contrast: $\tau = \varepsilon_r - 1 = 2.0 \pm 0.3$

```
         y
         ↑
         │
    ─────┼────●────────→ x
         │   (25mm, 0)
         │    ○ r=15mm
```

### Two Dielectric Cylinders

**Specifications:**
- Both: radius 15 mm, $\varepsilon_r = 3.0 \pm 0.3$
- Left cylinder: $(x, y) = (-45 \text{ mm}, +15 \text{ mm})$
- Right cylinder: $(x, y) = (+45 \text{ mm}, +5 \text{ mm})$
- Separation: 90 mm center-to-center

```
         y
         ↑
         │   ○ (-45mm, 15mm)
    ─────┼──────────────────→ x
         │              ○ (45mm, 5mm)
```

### Ground Truth Uncertainty

**Important:** The "ground truth" profile in the code is an **idealization**. Real targets have:
- Machining tolerances on dimensions
- Position uncertainty ($\sim 1$ mm)
- Permittivity measured with $\pm 10\%$ accuracy

This means NMSE $> 0$ even for a "perfect" reconstruction!

---

## 6. Exercises and Experiments

### Exercise 1: Single Cylinder Reconstruction

**Goal:** Validate algorithm on real data

**Steps:**
1. Load `dielTM_dec8f.txt` data
2. Run inversion, find optimal threshold
3. Compare reconstruction to nominal profile

**Questions:**
- What truncation threshold works best?
- Is the reconstructed contrast close to $\tau = 2$?
- Is the position accurately recovered?
- How does NMSE compare to simulated data?

### Exercise 2: Cross-Section Validation

**Goal:** Quantitative comparison to specifications

**Analysis:**
1. Extract x-cut through object center ($y = y_0$)
2. Extract y-cut through object center ($x = x_0$)
3. Compare reconstructed profile to:
   - Nominal value ($\varepsilon_r = 3$, so $\tau = 2$)
   - Tolerance band ($\varepsilon_r = 2.7$ to $3.3$, so $\tau = 1.7$ to $2.3$)

**Plot:**
```
ε_r
 │
3.3├─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Upper tolerance
 │        ╭──────╮
3.0├──────│ rec. │────── Nominal
 │        ╰──────╯
2.7├─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Lower tolerance
 │
1.0├──────────────────── Background
 └────────────────────────────► x
```

**Success criterion:** Peak of reconstruction falls within tolerance band

### Exercise 3: Two-Cylinder Resolution

**Goal:** Test resolution with real data

**Steps:**
1. Load `twodielTM_8f.txt` data
2. Run inversion
3. Check if both cylinders are resolved

**Questions:**
- Are both cylinders clearly visible?
- Is the separation correctly recovered (90 mm)?
- Are their positions accurate?
- Is contrast similar for both?

### Exercise 4: Frequency Dependence (Advanced)

**Goal:** Understand multi-frequency potential

**Background:** The Fresnel dataset includes multiple frequencies (2, 3, 4, ..., 8 GHz in the `_8f` files).

**Exploration:**
1. The current code uses 4 GHz
2. The data files contain multi-frequency data
3. Different frequencies give different resolution/penetration

**Note:** Modifying for different frequencies requires understanding the data format in `load_data_fr2001.p`.

### Exercise 5: Compare Simulated vs Experimental

**Goal:** Quantify simulation-to-reality gap

**Steps:**
1. Create simulated scenario matching Fresnel geometry:
   - Same object (15mm cylinder, $\tau = 2$)
   - Same position (25mm, 0)
   - Same frequency (4 GHz)
   - Same grid ($64 \times 64$)
   - Same measurement geometry

2. Run inversion on both with same threshold

3. Compare:

   | Metric | Simulated | Experimental |
   |--------|-----------|--------------|
   | Optimal threshold | | |
   | NMSE | | |
   | Peak contrast | | |
   | Position error | | |

**Expected:** Experimental NMSE slightly higher due to:
- Calibration imperfections
- Non-ideal noise characteristics
- Model approximations

---

## 7. Validation and Benchmarking

### What Makes a Good Reconstruction?

For the Fresnel data, assess:

1. **Position accuracy**
   - Is center within 1-2 mm of nominal?
   - For two cylinders, is separation correct?

2. **Size accuracy**
   - Is apparent radius close to 15 mm?
   - (Note: reconstruction resolution limits may blur edges)

3. **Contrast accuracy**
   - Is peak $\tau$ within tolerance (1.7-2.3)?
   - Is background close to zero?

4. **Shape quality**
   - Is it approximately circular?
   - Are artifacts minimal?

### Benchmarking Against Literature

The Fresnel dataset is widely used. Your results can be compared to:

| Method | Typical NMSE (single cyl.) | Notes |
|--------|---------------------------|-------|
| Born + TSVD | 0.15 - 0.30 | This exercise |
| Born Iterative | 0.10 - 0.20 | More iterations |
| Contrast Source Inv. | 0.05 - 0.15 | Handles nonlinearity |
| Deep Learning | 0.03 - 0.10 | Requires training |

**Your goal:** Achieve NMSE $< 0.3$ with clear visualization of the target

### Success Criteria Summary

**Single cylinder:**
- [ ] Object clearly visible in reconstruction
- [ ] Position within 5mm of (25mm, 0)
- [ ] Peak contrast $\tau_{\text{rec}} > 1.5$
- [ ] NMSE $< 0.35$

**Two cylinders:**
- [ ] Both objects clearly separated
- [ ] Separation approximately 90mm
- [ ] Both peaks have similar contrast
- [ ] NMSE $< 0.45$ (harder case)

---

## Summary: Experimental Data Insights

1. **Real data is messier** - expect higher NMSE than simulated
2. **Ground truth has uncertainty** - don't chase NMSE $= 0$
3. **Threshold choice may differ** - real noise isn't pure AWGN
4. **Visual validation matters** - check position, shape, not just NMSE
5. **This is how real imaging works** - valuable real-world experience

---

## References

1. Fresnel Institute Database: https://www.fresnel.fr/perso/geffrin/
2. Geffrin et al., "Free space experimental scattering database," *Inverse Problems*, 2005
3. Belkebir & Saillard, "Testing inversion algorithms against experimental data," *Inverse Problems*, 2001

---

*Next: See `05_QUICK_REFERENCE.md` for a condensed summary of key formulas and parameters.*
