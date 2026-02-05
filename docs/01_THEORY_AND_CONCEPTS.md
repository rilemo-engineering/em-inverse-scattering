# Theory and Concepts: Electromagnetic Inverse Scattering

This document provides the theoretical foundation for understanding the inverse scattering exercises in this repository.

---

## Table of Contents

1. [What is Inverse Scattering?](#1-what-is-inverse-scattering)
2. [The Physical Setup: 2D TM Configuration](#2-the-physical-setup-2d-tm-configuration)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Forward Problem vs Inverse Problem](#4-forward-problem-vs-inverse-problem)
5. [The Contrast Function](#5-the-contrast-function)
6. [The Born Approximation](#6-the-born-approximation)
7. [Ill-Posedness and Regularization](#7-ill-posedness-and-regularization)
8. [TSVD: Truncated Singular Value Decomposition](#8-tsvd-truncated-singular-value-decomposition)
9. [Degrees of Freedom and Resolution Limits](#9-degrees-of-freedom-and-resolution-limits)
10. [Quality Metrics](#10-quality-metrics)

---

## 1. What is Inverse Scattering?

**Inverse scattering** is the process of determining the properties (shape, location, material characteristics) of an unknown object by analyzing how it scatters electromagnetic waves.

### The Analogy
Think of it like this:
- **Forward Problem**: You know what object is in a dark room. You shine a flashlight at it and predict what shadow it casts.
- **Inverse Problem**: You see a shadow on the wall. You try to figure out what object caused it.

### Applications
- Medical imaging (microwave tomography)
- Non-destructive testing
- Ground-penetrating radar
- Security screening
- Geophysical exploration

---

## 2. The Physical Setup: 2D TM Configuration

### Why 2D?
We consider **two-dimensional** problems where:
- Objects are infinitely long cylinders along the $z$-axis
- All variations occur only in the $x$-$y$ plane
- This simplifies the mathematics while preserving essential physics

### TM Polarization (Transverse Magnetic)
In TM polarization:
- Electric field $\mathbf{E}$ is polarized along the $z$-axis: $\mathbf{E} = E_z(x,y) \hat{\mathbf{z}}$
- Magnetic field $\mathbf{H}$ lies in the $x$-$y$ plane
- This reduces Maxwell's vector equations to a single scalar equation

### The Measurement Geometry

```
                    Measurement Circle (radius Rm)
                         ╭─────────────╮
                       ╱               ╲
                      │    ┌─────────┐   │ ← Receivers (Nm points)
           Tx ────────│────│   DoI   │───│
         (Transmitter)│    │ (Object)│   │
                      │    └─────────┘   │
                       ╲               ╱
                        ╰─────────────╯
                               ↑
                    Investigation Domain (lx × ly)
```

**Key Elements:**
- **DoI (Domain of Investigation)**: Square region where the unknown object might be located
- **Transmitters (Tx)**: Sources that illuminate the object from different directions ($N_v$ views)
- **Receivers (Rx)**: Sensors that measure the scattered field ($N_m$ points)
- **Measurement circle**: Typically surrounds the DoI at radius $R_m$

---

## 3. Mathematical Foundations

### The Helmholtz Equation
For time-harmonic fields ($e^{-i\omega t}$ convention), the scalar wave equation becomes:

$$\nabla^2 E(\mathbf{r}) + k^2(\mathbf{r}) E(\mathbf{r}) = 0$$

where $k(\mathbf{r}) = \omega\sqrt{\varepsilon(\mathbf{r})\mu_0}$ is the spatially-varying wavenumber.

### The Lippmann-Schwinger Equation
The total field $E_{\text{tot}}$ satisfies the integral equation:

$$E_{\text{tot}}(\mathbf{r}) = E_{\text{inc}}(\mathbf{r}) + k_0^2 \iint_{\text{DoI}} \tau(\mathbf{r}') \cdot E_{\text{tot}}(\mathbf{r}') \cdot G(\mathbf{r},\mathbf{r}') \, d\mathbf{r}'$$

where:
- $E_{\text{inc}}(\mathbf{r})$: Incident field (known)
- $E_{\text{tot}}(\mathbf{r})$: Total field (unknown inside DoI)
- $\tau(\mathbf{r}')$: Contrast function (what we want to find!)
- $G(\mathbf{r},\mathbf{r}')$: Green's function (propagator)

### The Green's Function (2D)
For a homogeneous background, the 2D Green's function is:

$$G(\mathbf{r},\mathbf{r}') = \frac{i}{4} H_0^{(1)}(k_b|\mathbf{r} - \mathbf{r}'|)$$

where $H_0^{(1)}$ is the zeroth-order Hankel function of the first kind.

---

## 4. Forward Problem vs Inverse Problem

### Forward Problem
**Given**: Object properties ($\tau$), incident field ($E_{\text{inc}}$)
**Find**: Scattered field ($E_{\text{scat}}$)

This is a **well-posed problem**:
- Solution exists
- Solution is unique
- Solution depends continuously on the data

The forward solver in these exercises uses the **CGFFT method** (Conjugate Gradient with Fast Fourier Transform) to efficiently solve the Lippmann-Schwinger equation.

### Inverse Problem
**Given**: Scattered field ($E_{\text{scat}}$), incident field ($E_{\text{inc}}$)
**Find**: Object properties ($\tau$)

This is an **ill-posed problem**:
- Small errors in data cause large errors in reconstruction
- Multiple objects might produce similar scattered fields
- Requires regularization to obtain stable solutions

---

## 5. The Contrast Function

The **contrast function** $\tau(\mathbf{r})$ encodes the electromagnetic properties of the object:

$$\tau(\mathbf{r}) = \varepsilon_r(\mathbf{r}) - \varepsilon_b - i\frac{\sigma(\mathbf{r})}{\omega\varepsilon_0}$$

where:
- $\varepsilon_r(\mathbf{r})$: Relative permittivity of the object
- $\varepsilon_b$: Background permittivity (usually 1 for free space)
- $\sigma(\mathbf{r})$: Conductivity [S/m]
- $\omega$: Angular frequency [rad/s]
- $\varepsilon_0 = 8.85 \times 10^{-12}$ F/m: Vacuum permittivity

### Physical Interpretation
| $\tau$ value | Physical meaning |
|---------|------------------|
| $\tau = 0$ | Same as background (no scatterer) |
| $\text{Re}(\tau) > 0$ | Higher permittivity than background |
| $\text{Re}(\tau) < 0$ | Lower permittivity than background |
| $\text{Im}(\tau) < 0$ | Lossy material (energy absorption) |

### Example
For a dielectric object with $\varepsilon_r = 3$ in free space ($\varepsilon_b = 1$), $\sigma = 0$:

$$\tau = 3 - 1 - 0 = 2$$

---

## 6. The Born Approximation

### The Problem
The inverse problem is **nonlinear** because $E_{\text{tot}}$ depends on $\tau$, and we need $E_{\text{tot}}$ to find $\tau$.

### The Born Approximation (BA)
We linearize by assuming **weak scattering**:

$$E_{\text{tot}}(\mathbf{r}) \approx E_{\text{inc}}(\mathbf{r}) \quad \text{(inside the DoI)}$$

This assumes:
- The object is small
- The contrast is weak ($|\tau| \ll 1$)
- Multiple scattering is negligible

### The Linearized Problem
With the Born approximation, the scattered field becomes:

$$E_{\text{scat}}(\mathbf{r}_m) = k_0^2 \iint_{\text{DoI}} \tau(\mathbf{r}') \cdot E_{\text{inc}}(\mathbf{r}') \cdot G(\mathbf{r}_m,\mathbf{r}') \, d\mathbf{r}'$$

This is a **linear integral equation** in $\tau$!

### Matrix Form
After discretization:

$$\mathbf{E}_{\text{scat}} = \mathbf{S} \cdot \boldsymbol{\tau}$$

where:
- $\mathbf{E}_{\text{scat}}$: Vector of measured scattered fields $(N_m \cdot N_v \times 1)$
- $\mathbf{S}$: Scattering operator matrix $(N_m \cdot N_v \times N_x \cdot N_y)$
- $\boldsymbol{\tau}$: Vectorized contrast function $(N_x \cdot N_y \times 1)$

### When Does Born Fail?
The Born approximation breaks down when:
- Objects are electrically large (diameter $\gg \lambda$)
- Contrast is high ($|\tau| \geq 1$)
- Multiple scattering is significant

In these cases, iterative methods (Born Iterative Method, Contrast Source Inversion) are needed.

---

## 7. Ill-Posedness and Regularization

### What Makes the Problem Ill-Posed?

The scattering operator $\mathbf{S}$ has **singular values that decay rapidly**:

$$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \to 0$$

This means:
- Many components of $\tau$ produce negligible scattered fields
- Small noise gets amplified by $1/\sigma_i$ for small singular values
- The naive solution blows up!

### The Picard Condition
A solution exists only if the data coefficients $|\mathbf{u}_i^T \mathbf{b}|$ decay faster than the singular values $\sigma_i$.

With noisy data, this condition is violated for large $i$ (small $\sigma_i$).

### Regularization Strategy
**Goal**: Trade off between:
- **Data fidelity**: Match the measured data
- **Stability**: Avoid noise amplification

Common approaches:
- TSVD (used in these exercises)
- Tikhonov regularization
- Total Variation
- Sparsity constraints

---

## 8. TSVD: Truncated Singular Value Decomposition

### SVD Decomposition
The scattering operator $\mathbf{S}$ can be decomposed:

$$\mathbf{S} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U}$: Left singular vectors (measurement space)
- $\boldsymbol{\Sigma}$: Diagonal matrix of singular values
- $\mathbf{V}$: Right singular vectors (object space)

### The Naive (Unstable) Solution

$$\boldsymbol{\tau} = \mathbf{S}^+ \mathbf{E}_{\text{scat}} = \mathbf{V} \boldsymbol{\Sigma}^{-1} \mathbf{U}^T \mathbf{E}_{\text{scat}} = \sum_{i=1}^{n} \frac{\mathbf{u}_i^T \mathbf{E}_{\text{scat}}}{\sigma_i} \mathbf{v}_i$$

For small $\sigma_i$, the term $\frac{\mathbf{u}_i^T \mathbf{E}_{\text{scat}}}{\sigma_i}$ explodes due to noise!

### TSVD Solution
**Truncate** the expansion at index $k$:

$$\boldsymbol{\tau}_{\text{TSVD}} = \sum_{i=1}^{k} \frac{\mathbf{u}_i^T \mathbf{E}_{\text{scat}}}{\sigma_i} \mathbf{v}_i$$

This discards components associated with small singular values.

### Choosing the Truncation Index

The truncation threshold in dB is related to the truncation index $k$:

$$\text{Threshold [dB]} = 20 \cdot \log_{10}\left(\frac{\sigma_k}{\sigma_1}\right)$$

**Guidelines:**
| Threshold | Effect |
|-----------|--------|
| $-10$ dB | Very aggressive truncation, smooth result, low resolution |
| $-20$ dB | Moderate truncation, balanced |
| $-30$ dB | Mild truncation, higher resolution but more noise |
| $-40$ dB | Light truncation, prone to noise artifacts |

**Rule of thumb**: Choose threshold $\approx -\text{SNR}$ (in dB)

### The Bias-Variance Trade-off

```
        High truncation              Low truncation
        (small k)                    (large k)
        ─────────────────────────────────────────────►

        ↑ Bias (smoothing)           ↓ Bias
        ↓ Variance (noise)           ↑ Variance (noise)

        Blurry but stable            Sharp but noisy
```

---

## 9. Degrees of Freedom and Resolution Limits

### Degrees of Freedom (DoF)
The **information content** of the scattered field data is limited:

$$\text{DoF} \approx 2 \beta a$$

where:
- $\beta = k_0 = \frac{2\pi}{\lambda}$: Wavenumber
- $a$: Characteristic size of the object (e.g., $a = \frac{\sqrt{2} \cdot l_x}{2}$ for a square DoI)

### Physical Interpretation
- DoF determines the maximum number of resolvable pixels
- Cannot recover more independent information than the data contains
- Typical values: DoF $\approx 50$-$200$ for microwave imaging

### Resolution Limit
The **Rayleigh resolution limit** in inverse scattering:

$$\delta \approx \frac{\lambda}{2 \cdot \text{NA}}$$

where NA is the numerical aperture (related to angular coverage).

For full 360° coverage: $\delta \approx \frac{\lambda}{2}$ (diffraction limit)

### Aspect-Limited Configuration
When receivers/transmitters cover only a portion of the circle:
- Reduced angular diversity
- Anisotropic resolution (better in some directions)
- Artifacts in reconstruction

---

## 10. Quality Metrics

### Normalized Mean Square Error (NMSE)

$$\text{NMSE} = \frac{\|\tau_{\text{true}} - \tau_{\text{rec}}\|^2}{\|\tau_{\text{true}}\|^2} = \frac{\sum_{i,j}|\tau_{\text{true}}(i,j) - \tau_{\text{rec}}(i,j)|^2}{\sum_{i,j}|\tau_{\text{true}}(i,j)|^2}$$

| NMSE Value | Interpretation |
|------------|----------------|
| $0$ | Perfect reconstruction |
| $0.01$-$0.1$ | Good reconstruction |
| $0.1$-$0.5$ | Moderate quality |
| $> 0.5$ | Poor reconstruction |
| $1$ | Reconstruction as bad as zero |

### Visual Assessment
- **Position accuracy**: Is the object in the right place?
- **Shape fidelity**: Is the shape well-recovered?
- **Contrast accuracy**: Are the values correct?
- **Artifacts**: Are there spurious features?

---

## Summary: The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INVERSE SCATTERING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHYSICAL WORLD          MATHEMATICS           COMPUTATION       │
│  ─────────────          ───────────           ───────────        │
│                                                                  │
│  Unknown Object  ──►  Contrast τ(r)                              │
│       ↓                    ↓                                     │
│  Illumination    ──►  E_inc (known)                              │
│       ↓                    ↓                                     │
│  Scattering      ──►  Lippmann-Schwinger    ──►  Forward Solver  │
│       ↓                    ↓                       (CGFFT)       │
│  Measurement     ──►  E_scat (data)                              │
│       ↓                    ↓                                     │
│  Add Noise       ──►  E_scat + noise                             │
│       ↓                    ↓                                     │
│  Inversion       ──►  Born Approx: Sτ=E     ──►  TSVD Solver     │
│       ↓                    ↓                                     │
│  Reconstruction  ──►  τ_reconstructed                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

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

---

## Further Reading

1. **Colton & Kress** - "Inverse Acoustic and Electromagnetic Scattering Theory" (Springer)
2. **Chew** - "Waves and Fields in Inhomogeneous Media" (IEEE Press)
3. **Bucci & Isernia** - "Electromagnetic Inverse Scattering: Retrievable Information and Measurement Strategies" (Radio Science, 1997)
4. **Hansen** - "Rank-Deficient and Discrete Ill-Posed Problems" (SIAM)

---

*Next: See `02_FORWARD_PROBLEM_GUIDE.md` for hands-on exercise instructions.*
