# Exercise Guide: Forward Scattering Problem

This guide walks you through the forward scattering exercises in `matlab/simulated/scenario/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Before You Start](#2-before-you-start)
3. [Running the Forward Solver](#3-running-the-forward-solver)
4. [Understanding the Outputs](#4-understanding-the-outputs)
5. [Visualization Walkthrough](#5-visualization-walkthrough)
6. [Exercises and Experiments](#6-exercises-and-experiments)
7. [What to Observe and Learn](#7-what-to-observe-and-learn)

---

## 1. Overview

### What is the Forward Problem?

The forward problem answers: **"Given an object, what scattered field does it produce?"**

```
INPUT:  Object profile τ(x,y), incident fields E_inc
        ↓
        Forward Solver (CGFFT iterative method)
        ↓
OUTPUT: Scattered field data E_scat(receiver, transmitter)
```

### Why Study the Forward Problem First?

1. **Understanding data**: Before inverting, you must understand what scattered fields look like
2. **Building intuition**: How do object properties affect the scattered field?
3. **Generating test data**: Create synthetic data with known "ground truth" for testing inversion

---

## 2. Before You Start

### Prerequisites
- MATLAB (R2018b or later recommended)
- Basic understanding of complex numbers and matrix operations
- Read `01_THEORY_AND_CONCEPTS.md` first

### File Structure
```
matlab/simulated/scenario/
├── c1_Scenario.m          ← Main script (EDIT THIS)
├── forward_solver.p       ← Forward solver (protected)
├── CGFFT.p               ← FFT-accelerated solver
├── ainterno.p            ← Internal field calculation
├── Profili.p             ← Profile generation
└── DATA_scenario.mat     ← Output data file
```

### Key Parameters to Know

| Parameter | In Code | Description | Typical Value |
|-----------|---------|-------------|---------------|
| Frequency | `freq` | Working frequency [Hz] | 300 MHz - 10 GHz |
| DoI size | `lx`, `ly` | Investigation domain [m] | 0.1 - 1.0 m |
| Grid size | $N_x$, $N_y$ | Discretization points | 32 - 128 |
| Meas. radius | $R_m$ | Measurement circle radius [m] | $> l_x\sqrt{2}/2$ |
| # Receivers | $N_m$ | Number of measurement points | $\approx$ DoF |
| # Transmitters | $N_v$ | Number of illumination angles | $\approx N_m$ |
| Background $\varepsilon$ | `eb` | Background permittivity | 1 (free space) |
| Background $\sigma$ | `sb` | Background conductivity [S/m] | 0 |
| Object $\varepsilon$ | `ex` | Object permittivity | 1.5 - 10 |
| Object $\sigma$ | `sx` | Object conductivity [S/m] | 0 - 1 |

---

## 3. Running the Forward Solver

### Step-by-Step Instructions

1. **Open MATLAB** and navigate to `matlab/simulated/scenario/`

2. **Open `c1_Scenario.m`** - Note the important comments at the top:
   ```matlab
   % BEFORE RUNNING THE CODE, TAKE NOTE ABOUT USEFUL DATA INFORMATION
   % - freq: working frequency [MHz]
   % - eb: permittivity of the background
   % - sb: conductivity of the background
   % - lx: x-dimension of the Domain of Investigation [m]
   % ...
   ```

3. **Set the number of iterations**:
   ```matlab
   n_iter = 1000;  % Number of CGFFT iterations
   ```
   - Higher values = more accurate but slower
   - 500-1000 is usually sufficient for convergence

4. **Run the script**:
   ```matlab
   >> c1_Scenario
   ```

5. **Answer the visualization prompt**:
   ```
   Would you like to visualize fields and profile? Yes[1] No[0]:
   ```
   Type `1` and press Enter to see the plots.

### What Happens Internally

```
forward_solver.p
    │
    ├─► Profili.p          → Creates object profile τ(x,y)
    │
    ├─► ainterno.p         → Computes internal field operator
    │
    ├─► CGFFT.p            → Iteratively solves for E_tot
    │                         (Conjugate Gradient + FFT acceleration)
    │
    └─► Computes E_scat at measurement points
```

---

## 4. Understanding the Outputs

### Variables Saved in `DATA_scenario.mat`

| Variable | Size | Description |
|----------|------|-------------|
| `Escat` | $N_m \times N_v$ | **Scattered field data matrix** |
| `PROF` | $N_y \times N_x$ | **Contrast profile** $\tau(x,y)$ |
| `Einc_domain` | $N_y \times N_x \times N_v$ | Incident field on DoI for each view |
| `Etot_domain` | $N_y \times N_x \times N_v$ | Total field on DoI for each view |
| `freq` | scalar | Frequency [Hz] |
| `lx`, `ly` | scalar | DoI dimensions [m] |
| $N_x$, $N_y$ | scalar | Grid points |
| $\varepsilon_b$, $\sigma_b$ | scalar | Background properties |
| $R_m$ | scalar | Measurement radius [m] |
| `DOF` | scalar | Degrees of freedom |

### The MVMS Data Matrix (Escat)

**MVMS** = Multi-View Multi-Static

```
         Transmitter index (v = 1, 2, ..., Nv)
              ↓
         ┌─────────────────────────┐
         │                         │
    Rx 1 │  E_scat(1,1)  ...       │
    Rx 2 │  E_scat(2,1)  ...       │  ← Each column: scattered field
     ⋮   │      ⋮         ⋱        │     for all receivers due to
    Rx Nm│  E_scat(Nm,1) ...       │     one transmitter
         │                         │
         └─────────────────────────┘

    Each entry E_scat(m,v) = scattered field at receiver m
                             due to transmitter v
```

---

## 5. Visualization Walkthrough

### Figure 1: Scenario Overview

```
┌────────────────────────────────────────┐
│                                        │
│         ● ● ● ● ●                      │  ← Measurement points (red dots)
│       ●           ●                    │
│      ●   ┌─────┐   ●                   │  ← DoI boundary (black square)
│     ●    │█████│    ●                  │  ← Object (grayscale)
│      ●   │█████│   ●                   │
│       ●   └─────┘  ●                   │
│         ● ● ● ● ●                      │
│                                        │
│    - - - - - - - - - - - - - -         │  ← Measurement circle (dashed)
│                                        │
└────────────────────────────────────────┘
```

**What to check:**
- Is the object inside the DoI?
- Are measurement points outside the DoI?
- Is the measurement circle large enough?

### Figure 2: Contrast Profile

Two subplots showing:
- **Left**: $\text{Re}[\tau]$ - Real part (related to permittivity contrast)
- **Right**: $\text{Im}[\tau]$ - Imaginary part (related to conductivity/losses)

**What to observe:**
- Shape and location of the object
- Magnitude of contrast ($\tau = 2$ means $\varepsilon_r = 3$)
- Axes normalized to wavelength ($x/\lambda_0$, $y/\lambda_0$)

### Figure 3: MVMS Data Matrix

Shows $|E_{\text{scat}}|$ as an image:
- **x-axis**: Transmitter index ($n_v$)
- **y-axis**: Receiver index ($n_m$)

**What to observe:**
- Diagonal structure (strongest scattering in forward/backscatter directions)
- Symmetry patterns
- Magnitude variations

### Figure 4: Incident Field Animation

For each transmitter position:
- **Left**: $|E_{\text{inc}}|$ amplitude
- **Right**: Phase of $E_{\text{inc}}$

**What to observe:**
- Plane wave or cylindrical wave illumination
- Phase fronts rotating as transmitter moves around the object

---

## 6. Exercises and Experiments

### Exercise 1: Understand the Effect of Frequency

**Goal**: See how wavelength affects the scattered field

**Steps**:
1. Run with default frequency (note $\lambda_0 = c/f$)
2. Open `forward_solver.p` won't work (protected), but you can modify the forward solver call or find where freq is set
3. Double the frequency → wavelength halves
4. Compare MVMS matrices

**Questions to answer**:
- How does the MVMS matrix change with frequency?
- What happens to the DoF?
- Is the object more or less "visible"?

### Exercise 2: Vary Object Contrast

**Goal**: Understand weak vs strong scattering

**Steps**:
1. Default: $\tau = 2$ ($\varepsilon_r = 3$) → moderate contrast
2. Try $\tau = 0.5$ ($\varepsilon_r = 1.5$) → weak scattering (Born valid)
3. Try $\tau = 5$ ($\varepsilon_r = 6$) → strong scattering (Born breaks down)

**Questions to answer**:
- How does $|E_{\text{scat}}|$ scale with $\tau$ for weak contrast?
- At what contrast does linearity break down?

### Exercise 3: Object Size and Position

**Goal**: Understand resolution limits

**Steps**:
1. Try a small object (radius $< \lambda/4$)
2. Try a large object (radius $> \lambda$)
3. Move object to different positions

**Questions to answer**:
- When is the object well-resolved?
- How does position affect the scattered field pattern?

### Exercise 4: Multiple Objects

**Goal**: Understand interactions

**Steps**:
1. Create two separate objects
2. Vary separation distance

**Questions to answer**:
- Can you distinguish two objects in the MVMS data?
- At what separation do they "merge"?

### Exercise 5: Number of Measurements

**Goal**: Understand sampling requirements

**Steps**:
1. Default: $N_m \approx$ DoF
2. Reduce $N_m$ to DoF/2
3. Increase $N_m$ to $2 \times$ DoF

**Questions to answer**:
- What is the minimum $N_m$ for good data?
- Is there benefit to oversampling?

---

## 7. What to Observe and Learn

### Key Physical Insights

1. **Wavelength is the ruler**
   - Objects smaller than $\lambda/2$ are hard to resolve
   - Plot axes in units of $\lambda_0$ for meaningful comparison

2. **MVMS matrix structure**
   - Diagonal dominance → forward/backward scattering
   - Off-diagonal → multi-static information
   - Symmetries reflect object symmetries

3. **Field behavior**
   - Inside object: $E_{\text{tot}} \neq E_{\text{inc}}$ (contrast causes modification)
   - Strong contrast → more multiple scattering → harder to invert

4. **Degrees of Freedom**

   $$\text{DoF} = 2 \cdot \frac{2\pi}{\lambda} \cdot \frac{\sqrt{2} \cdot l_x}{2} \approx \frac{2\pi\sqrt{2} \cdot l_x}{\lambda}$$

   - This limits how many independent "pixels" you can reconstruct
   - More DoF with higher frequency or larger DoI

### Checklist Before Moving to Inversion

Before proceeding to `inversion/`:

- [ ] I understand what PROF represents (contrast function $\tau$)
- [ ] I understand the MVMS data matrix structure
- [ ] I can relate wavelength to object size
- [ ] I know what DoF means and how it's calculated
- [ ] I've observed how $E_{\text{inc}}$ changes with transmitter position
- [ ] I've saved `DATA_scenario.mat` to use in inversion

---

## Common Issues and Troubleshooting

### Issue: "Cannot find forward_solver"
**Solution**: Make sure you're in the correct directory (`matlab/simulated/scenario/`)

### Issue: Slow convergence
**Cause**: High contrast or large object
**Solution**: Increase `n_iter` or reduce object contrast

### Issue: Strange artifacts in profile
**Cause**: Discretization too coarse
**Solution**: Increase $N_x$, $N_y$ (but this slows computation)

### Issue: Memory error
**Cause**: Grid too large
**Solution**: Reduce $N_x$, $N_y$ or close other applications

---

*Next: See `03_INVERSE_PROBLEM_GUIDE.md` for the reconstruction exercises.*
