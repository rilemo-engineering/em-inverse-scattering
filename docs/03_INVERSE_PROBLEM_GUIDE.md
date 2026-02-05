# Exercise Guide: Inverse Scattering Problem

This guide walks you through the inverse scattering exercises in `matlab/simulated/inversion/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Before You Start](#2-before-you-start)
3. [Running the Inversion](#3-running-the-inversion)
4. [The Critical Choice: Truncation Threshold](#4-the-critical-choice-truncation-threshold)
5. [Understanding the Outputs](#5-understanding-the-outputs)
6. [Noise Effects](#6-noise-effects)
7. [Aspect-Limited Configuration](#7-aspect-limited-configuration)
8. [Exercises and Experiments](#8-exercises-and-experiments)
9. [Troubleshooting and Common Pitfalls](#9-troubleshooting-and-common-pitfalls)

---

## 1. Overview

### What is the Inverse Problem?

The inverse problem answers: **"Given scattered field measurements, what object produced them?"**

```
INPUT:  Scattered field data E_scat, Incident fields E_inc
        ↓
        Born Approximation: E_scat ≈ S · τ
        ↓
        TSVD Regularization
        ↓
OUTPUT: Reconstructed profile τ_rec(x,y)
```

### The Inversion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. Load DATA_scenario.mat (from forward solver)                 │
│                    ↓                                             │
│  2. Add noise: E_scat_noisy = awgn(E_scat, SNR)                 │
│                    ↓                                             │
│  3. Born approximation: E_tot ≈ E_inc inside DoI                 │
│                    ↓                                             │
│  4. Build scattering operator S = kernel_scattering(...)         │
│                    ↓                                             │
│  5. Compute SVD: S = U · Σ · V'                                  │
│                    ↓                                             │
│  6. Plot singular value spectrum                                 │
│                    ↓                                             │
│  7. USER CHOOSES truncation threshold [dB]                       │
│                    ↓                                             │
│  8. Solve with TSVD: τ_rec = TSVD_solver(U, Σ, V, k, E_scat)    │
│                    ↓                                             │
│  9. Compare τ_rec with τ_true, compute NMSE                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Before You Start

### Prerequisites
- Complete the forward problem exercises first
- Have `DATA_scenario.mat` ready (from `matlab/simulated/scenario/`)
- Understand SVD basics (see Theory document)

### File Structure
```
matlab/simulated/inversion/
├── c2_Inversion_BORN.m     ← Main script (EDIT THIS)
├── kernel_scattering.p     ← Builds scattering operator
├── TSVD_solver.p          ← TSVD reconstruction
├── DATA_scenario.mat      ← Input (copy from scenario/)
├── DATA_scenario_square.mat    ← Alternative: square object
└── DATA_scenario_noweak.mat    ← Alternative: strong scatterer
```

### Available Datasets

| Dataset | Description | Born Valid? |
|---------|-------------|-------------|
| `DATA_scenario.mat` | Default circular target | Yes |
| `DATA_scenario_square.mat` | Square target | Yes |
| `DATA_scenario_noweak.mat` | Strong scatterer | **No** |

---

## 3. Running the Inversion

### Step-by-Step Instructions

1. **Copy data file** from `scenario/`:
   ```matlab
   % In inversion folder
   copyfile('../scenario/DATA_scenario.mat', '.')
   ```

2. **Open `c2_Inversion_BORN.m`** and review the settings:
   ```matlab
   load DATA_scenario.mat    % Choose which dataset
   AL = 0;                   % 0 = full aspect, 1 = aspect-limited
   SNR = 30;                 % Signal-to-noise ratio [dB]
   ```

3. **Run the script**:
   ```matlab
   >> c2_Inversion_BORN
   ```

4. **Observe the singular value plot** (Figure 1):
   - Y-axis: Normalized singular values in dB
   - X-axis: Singular value index

5. **Enter truncation threshold** when prompted:
   ```
   Truncation threshold [dB]: -25
   ```

6. **Examine the reconstruction** (Figures 2-3)

---

## 4. The Critical Choice: Truncation Threshold

### Understanding the Singular Value Spectrum

```
    0 ┬─────────────────────────────────
      │●
      │ ●●
  -20 ┤   ●●●
      │      ●●●●●                    ← "Knee" of the curve
  -40 ┤           ●●●●●●●●
      │                    ●●●●●●●●●●●●  ← Noise-dominated region
  -60 ┤
      │
  -80 ┴─────────────────────────────────
      0        50       100      150
              Singular Value Index
```

### How to Choose

**Method 1: Visual (L-curve heuristic)**
- Look for the "knee" where spectrum transitions from signal to noise
- Choose threshold just before the noise floor

**Method 2: Noise-based**
- If SNR $= 30$ dB, try threshold $\approx -25$ to $-30$ dB
- Rule: threshold $\approx -(\text{SNR} - 5)$ dB

**Method 3: Trial and error**
- Start conservative (e.g., $-20$ dB)
- Increase magnitude ($-25$, $-30$, ...) until artifacts appear

### Effect of Threshold Choice

| Threshold | Truncation Index $k$ | Result |
|-----------|-------------------|--------|
| $-10$ dB | Small $k$ (~20) | Very blurry, underfit |
| $-20$ dB | Medium $k$ (~50) | Smooth, reasonable |
| $-30$ dB | Large $k$ (~100) | Sharper, some noise |
| $-40$ dB | Very large $k$ | Sharp but noisy |
| $-50$ dB | Near full rank | Noise explosion |

### Visual Comparison

```
Threshold: -15 dB         Threshold: -30 dB         Threshold: -45 dB
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│             │           │     ██      │           │ ░░  ██  ░░  │
│    ░░░░     │           │    ████     │           │ ░░ ████ ░░░ │
│   ░░░░░░    │           │   ██████    │           │░░ ██████ ░░░│
│    ░░░░     │           │    ████     │           │ ░░ ████ ░░░ │
│             │           │     ██      │           │ ░░  ██  ░░  │
└─────────────┘           └─────────────┘           └─────────────┘
   Too smooth                Just right              Too noisy
   (High bias)              (Balanced)              (High variance)
```

---

## 5. Understanding the Outputs

### Figure 1: Singular Value Spectrum

**Elements:**
- Blue curve: Normalized singular values $20 \cdot \log_{10}(\sigma_i/\sigma_1)$
- Red dashed lines: Your chosen threshold and corresponding truncation index

**What to observe:**
- Smooth decay in signal region
- Level-off in noise region
- Number of singular values above threshold = effective DoF

### Figure 2: Reconstruction Comparison

Four-panel comparison:
```
┌────────────────┬────────────────┐
│ Re[τ]: actual  │ Im[τ]: actual  │
├────────────────┼────────────────┤
│ Re[τ]: recon.  │ Im[τ]: recon.  │
└────────────────┴────────────────┘
```

**What to compare:**
- Position: Is object in the right place?
- Shape: Are edges preserved?
- Magnitude: Are contrast values correct?
- Background: Is it clean ($\tau \approx 0$)?

### Figure 3: Normalized Reconstruction

- Grayscale: $|\tau_{\text{rec}}|/\max(|\tau_{\text{rec}}|)$
- Black contour: True object boundary

**Purpose:** Easier shape comparison regardless of absolute values

### NMSE Value

Printed in command window:
```matlab
NMSE_BORN = 0.1234
```

**Interpretation:**
- NMSE $< 0.1$: Good reconstruction
- NMSE $0.1$-$0.3$: Acceptable
- NMSE $> 0.5$: Poor

---

## 6. Noise Effects

### Adding Noise

In the code:
```matlab
SNR = 30;  % Signal-to-noise ratio in dB
Escat = awgn(Escat, SNR, 'measured', 345);
```

The `awgn` function adds white Gaussian noise:
- SNR $= 30$ dB → noise power $1000\times$ smaller than signal
- SNR $= 20$ dB → noise power $100\times$ smaller than signal
- SNR $= 10$ dB → noise power $10\times$ smaller than signal

### Experiment: Noise Level Sweep

Try these SNR values:

| SNR [dB] | Suggested Threshold | Expected NMSE |
|----------|-------------------|---------------|
| 40 | $-35$ dB | $< 0.05$ |
| 30 | $-25$ dB | $0.05$-$0.15$ |
| 20 | $-18$ dB | $0.15$-$0.30$ |
| 10 | $-10$ dB | $0.30$-$0.60$ |
| 0 | $-5$ dB | $> 0.60$ |

### What Happens with Noise

1. **High SNR (little noise):**
   - More singular values are "useful"
   - Can use lower (more negative) threshold
   - Sharper reconstruction

2. **Low SNR (much noise):**
   - Noise "fills in" small singular value components
   - Must truncate more aggressively
   - Blurrier reconstruction

---

## 7. Aspect-Limited Configuration

### What is Aspect Limitation?

In practice, you may not have receivers/transmitters surrounding the object completely (360°).

**Example:** Only upper arc coverage

```
Full Aspect:                    Aspect Limited:
    ● ● ● ● ●                       ● ● ● ● ●
  ●           ●                   ×           ×
 ●   [Object]   ●                ×   [Object]   ×
  ●           ●                   ×           ×
    ● ● ● ● ●                       × × × × ×

    ● = active                      ● = active
                                    × = inactive
```

### Enabling Aspect Limitation

In `c2_Inversion_BORN.m`:
```matlab
AL = 1;  % Set to 1 for aspect-limited
```

The mask in the code:
```matlab
mask1 = zeros(Nm, Nv);
mask1(2:10, 2:10) = 1;  % Only upper-left portion of data matrix
```

### Effects of Aspect Limitation

| Full Aspect | Aspect Limited |
|-------------|----------------|
| Isotropic resolution | Anisotropic resolution |
| More singular values | Fewer useful singular values |
| Complete angular info | Missing angular information |
| Good all-around | Good in covered directions |

### Exercise: Compare Full vs Limited

1. Run with `AL = 0`, note NMSE
2. Run with `AL = 1`, same threshold
3. Compare:
   - NMSE values
   - Reconstruction quality
   - Artifacts direction

---

## 8. Exercises and Experiments

### Exercise 1: Optimal Truncation (THE most important!)

**Goal:** Understand bias-variance trade-off

**Steps:**
1. Fix SNR $= 30$ dB
2. Run inversion with thresholds: $-10$, $-15$, $-20$, $-25$, $-30$, $-35$, $-40$ dB
3. Record NMSE for each
4. Plot NMSE vs threshold

**Expected result:**
```
NMSE
  │
  │ ●                           ●
  │  ●                        ●
  │   ●●                   ●●
  │     ●●●           ●●●●
  │        ●●●●●●●●●●●
  └────────────────────────────────
       -10    -20    -30    -40
            Threshold [dB]

   ↑ Under-regularized (noise)
                         ↑ Over-regularized (blur)
        OPTIMAL (minimum NMSE) ↑
```

### Exercise 2: Noise Sensitivity Study

**Goal:** Understand regularization under different noise levels

**Steps:**
1. Create a table:

   | SNR | Optimal Threshold | Min NMSE | Truncation Index $k$ |
   |-----|------------------|----------|------------------|
   | 40  |                  |          |                  |
   | 30  |                  |          |                  |
   | 20  |                  |          |                  |
   | 10  |                  |          |                  |

2. For each SNR, find optimal threshold by trial
3. Record results

**Analysis questions:**
- How does optimal threshold relate to SNR?
- How does minimum achievable NMSE depend on SNR?
- Is there a "rule of thumb" relationship?

### Exercise 3: Born Approximation Breakdown

**Goal:** See when linearization fails

**Steps:**
1. Load `DATA_scenario_noweak.mat` (strong scatterer)
2. Run inversion with various thresholds
3. Compare with original `DATA_scenario.mat`

**What to observe:**
- NMSE is higher even at optimal threshold
- Reconstruction shows systematic errors (not just noise)
- Shape may be distorted, not just blurred

**Why?** The Born approximation ($E_{\text{tot}} \approx E_{\text{inc}}$) is violated for strong scatterers.

### Exercise 4: Resolution Test

**Goal:** Understand resolution limits

**Steps:**
1. Create data with two close objects (in forward solver)
2. Vary separation: $2\lambda$, $\lambda$, $\lambda/2$, $\lambda/4$
3. For each, run inversion and check if objects are resolved

**Resolution criterion:** Can you see two distinct peaks?

### Exercise 5: Aspect Limitation Impact

**Goal:** Quantify information loss from limited aperture

**Steps:**
1. With full aspect (AL=0): record NMSE at optimal threshold
2. With aspect limited (AL=1): find new optimal threshold
3. Compare:
   - Optimal NMSE values
   - Required truncation indices
   - Reconstruction artifacts

**Questions:**
- How much does NMSE increase?
- What spatial features are lost?
- Is the degradation isotropic?

### Exercise 6: Different Object Shapes

**Goal:** See how shape affects reconstruction

**Steps:**
1. Compare `DATA_scenario.mat` (circular) vs `DATA_scenario_square.mat` (square)
2. Same SNR, find optimal threshold for each
3. Compare NMSE and visual quality

**Observations:**
- Are edges preserved equally?
- Which shape is "harder" to reconstruct?
- How do artifacts differ?

---

## 9. Troubleshooting and Common Pitfalls

### Pitfall 1: Using Too Low a Threshold

**Symptom:** Reconstruction explodes with noise

**Fix:** Use more aggressive truncation (higher dB value, e.g., $-20$ instead of $-40$)

### Pitfall 2: Using Too High a Threshold

**Symptom:** Reconstruction is a blob with no features

**Fix:** Use less aggressive truncation (lower dB value, e.g., $-30$ instead of $-15$)

### Pitfall 3: Mismatched Data

**Symptom:** Completely wrong reconstruction

**Cause:** Forward data parameters don't match inversion assumptions

**Fix:** Ensure same `freq`, `lx`, `ly`, $N_x$, $N_y$, $\varepsilon_b$, $\sigma_b$ in both

### Pitfall 4: Strong Scatterer with Born

**Symptom:** Systematic errors, wrong contrast magnitude

**Cause:** Born approximation invalid

**Limitation:** This codebase only supports Born approximation. For strong scatterers, you would need iterative methods (BIM, CSI) not included here.

### Pitfall 5: Not Enough Measurements

**Symptom:** Artifacts, aliasing patterns

**Cause:** $N_m <$ DoF (undersampled data)

**Fix:** Generate new forward data with more measurements

### Checklist: Good Inversion Practice

- [ ] Verified data loaded correctly (check variable sizes)
- [ ] Chose SNR realistic for the application
- [ ] Examined singular value spectrum before choosing threshold
- [ ] Started with conservative threshold, then refined
- [ ] Checked NMSE is in reasonable range
- [ ] Visually compared reconstruction to ground truth
- [ ] Understood any artifacts present

---

## Summary: Key Takeaways

1. **TSVD truncation is THE critical decision** - it determines reconstruction quality more than anything else

2. **Match truncation to noise level** - roughly threshold $\approx -(\text{SNR} - 5)$ dB as starting point

3. **Born approximation has limits** - works for weak scatterers ($|\tau| < 1$) only

4. **More data doesn't always help** - limited by DoF; oversampling adds redundancy, not information

5. **Aspect limitation costs resolution** - preferentially in directions not covered

6. **NMSE is informative but not complete** - also check visual quality and artifact patterns

---

*Next: See `04_EXPERIMENTAL_DATA_GUIDE.md` for working with real Fresnel data.*
