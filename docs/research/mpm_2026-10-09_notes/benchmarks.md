# Benchmarks and validation scenes for comparing two MPM / particle simulators (Isaac/Newton vs. Apple-Metal MPM)

As of October 2026. Researcher notes; every number has a link to its source. Anything recalled from memory without a fetched source sits under Gaps or is marked "(unverified)". Budget: about 15 search/fetch calls. Three primary PDFs (Lajeunesse 2005, Gao 2018, Li 2013) downloaded but would not convert to text here (no pdftotext/pypdf), so some coefficients come from secondary tables. Each such case is flagged.

## Granular column collapse (runout and final-height scaling vs aspect ratio)

### Takeaway
Column collapse is the best-documented quantitative granular benchmark. The normalized runout (L_f - L_i)/L_i against aspect ratio a = H_i/L_i follows a piecewise power law: linear for small a, then a^(2/3) in 2D or a^(1/2) for axisymmetric columns. Published prefactors and transition points vary by about 20-25% across labs (1.0 vs 1.2 linear prefactor; transition at a = 2.3 vs 3). Numerical papers usually claim agreement "with the trend" rather than quoting percent errors. A realistic acceptance band between two simulators is therefore roughly 10-20% on runout, with the fitted exponent matched.

### Cited Findings
- **Scaling laws (secondary table, quoted from an SPH column-collapse paper):** normalized runout (r_inf - r0)/r0 vs a = H0/r0 — [SPH granular column collapse study, arXiv 1602.07881](https://arxiv.org/html/1602.07881v1)
  - Lube et al. 2004, 3D axisymmetric: 1.24a for a < 1.7; 1.6a^(1/2) for a >= 1.7.
  - Lajeunesse et al. 2005, axisymmetric (semicircular tube): a for a < 3; a^(1/2) for a >= 3.
  - Lube et al. 2005, 2D between vertical walls: 1.2a for a < 2.3; 1.9a^(2/3) for a >= 2.3.
  - Lajeunesse et al. 2005, 2D channel: a for a < 3; a^(2/3) for a >= 3.
  - Staron & Hinch 2005, 2D DEM (numerical): 2.5a for a < 2; 3.25a^0.7 for a >= 2.
- The Lajeunesse paper is "Granular slumping on a horizontal surface", Lajeunesse, Monnier & Homsy, Phys. Fluids 17, 103302 (2005), DOI 10.1063/1.2085378. It used glass beads in a rectangular channel (2D) and a semicircular tube (axisymmetric) — [author PDF, IPGP](https://www.ipgp.fr/~lajeunes/download/2005-pof-slumping.pdf) (could not parse the text; coefficients above come from the secondary table)
- **SPH reproduction and its error:** 2D SPH gave 1.3a (a < 2) and 1.55a^(4/5) (a >= 2), overestimating runout at high a compared with Lube 2005. 3D SPH gave 0.85a (a < 1.7) and 1.05a^(3/5), underestimating it. No percent errors were reported, only qualitative "good agreement". Setup: phi = 30 deg (20/40 deg also tested), r0 = 9.7 cm, rho = 2.6 g/cm^3, smoothing ratio h/dr = 2 (2D) and 1.5625 (3D), h/L from 1/16 to 1/64, CFL prefactor 0.125 — [arXiv 1602.07881](https://arxiv.org/html/1602.07881v1)
- **Dunatunga & Kamrin 2015 (JFM 779:483-513):** an MPM code with a mu(I) elasto-viscoplastic model. It was validated against Beverloo scaling in silo flow, against "jointed power-law scaling of the run-out distance" in column collapse, and against inclined-chute behaviour. Heaps flatten as a increases, and the repose angle comes out below the static-friction value — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/107838?show=full); [arXiv 1411.5447](https://arxiv.org/pdf/1411.5447)
- A Mohr-Coulomb MPM study found MPM runout matches DEM for h/r < 2 but overpredicts it for h/r > 2. This is a common, known failure mode of rate-independent continuum models at high aspect ratio — summarized in search results pointing to [par.nsf.gov 10087501](https://par.nsf.gov/servlets/purl/10087501) (not fetched directly)
- Mast et al. (Acta Geotechnica, "Simulating granular column collapse using the Material Point Method") is listed in the Dunatunga & Kamrin references; its results were not retrieved — [arXiv 1411.5447](https://arxiv.org/pdf/1411.5447)

### Inferences
- Recommended scene for the two-simulator comparison: a quasi-2D slab (thin periodic or frictionless side walls) in which a varies over {0.5, 1, 2, 3, 5, 8} at fixed L_i (e.g., 0.1 m). Measure (L_f - L_i)/L_i, H_f/L_i, deposit profile h(x), time to arrest, and the KE(t) curve. Fit prefactor and exponent per simulator, then compare each fit with the other simulator and with Lube 2005 / Lajeunesse 2005.
- Because experimental prefactors themselves differ by about 20% (1.0 vs 1.2), agreement within about 10% between Newton and Metal, plus matching exponents, is a defensible pass. Matching experiment within 20% is "good" by the standard of published continuum papers (inference).
- Both Isaac/Newton and the Metal code need the same friction angle (Drucker-Prager), the same base friction and the same particles per cell. The runout is sensitive to base friction and to the high-a regime.

### Gaps
- Balmforth & Kerswell 2005 (JFM, "Granular collapse in two dimensions") coefficients and their channel-width dependence were not retrieved.
- Final-height laws were not retrieved from a fetched source. From memory (unverified), Lube 2005 gives H_f/L_i ~ a for a < ~1.15 and ~a^(2/5) beyond, and Lajeunesse gives H_f/H_i ~ a^-0.6 to a^-0.7.
- No percent errors were found for Mast 2014, Dunatunga & Kamrin 2015, Klár 2016 or Daviet & Bertails-Descoubes 2016 runout reproductions.

## Sand pile / angle of repose, silo discharge (Beverloo), RFT penetration (Li, Zhang & Goldman 2013)

### Takeaway
Beverloo discharge (W ∝ (D - kd)^(5/2), C ≈ 0.58, k ≈ 1.5) and the Li-Zhang-Goldman RFT coefficients are the two most quantitative, simulator-independent granular targets. RFT is directly robotics-relevant: it predicts plate and foot forces to within 10-35%. Angle of repose against Drucker-Prager friction angle is easy to measure, but published MPM papers treat it qualitatively and report that the repose angle comes out below the input friction angle.

### Cited Findings
- **Beverloo law:** W = C rho_b sqrt(g) (D0 - k d)^(5/2). Flow is independent of fill height. C ≈ 0.58 for almost all materials, with a reported band of 0.55-0.65 and up to 0.64 for glass spheres. k ≈ 1.5 for spheres (range 1.3-2.9, larger for angular grains). k ≈ 1.4 is a design default. The law is valid for d > 0.5 mm and orifices large enough to avoid jamming. For non-circular orifices, use the hydraulic diameter 4A/P — [Thermopedia](https://thermopedia.com/content/821/); [Purdue ME308 example](https://engineering.purdue.edu/~wassgren/teaching/ME30800/Examples/dim_anal_18.pdf); [arXiv 1611.05277](https://arxiv.org/pdf/1611.05277); [arXiv 2005.07042](https://arxiv.org/pdf/2005.07042); [powderprocess.net](https://www.powderprocess.net/Powder_Flow/DRAFT_Hopper_Discharge_Flowrate.html)
- Dunatunga & Kamrin's MPM recovers Beverloo scaling in silo flow — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/107838?show=full)
- **Klár et al. 2016 (SIGGRAPH), "Drucker-Prager Elastoplasticity for Sand Animation":** Hencky-strain hyperelasticity plus Drucker-Prager with a non-associative return mapping that works with explicit and implicit integration. They note that grain-level friction is hard to tune to match piling, and that Drucker-Prager lets one use the engineering literature to set parameters. Their dissertation has sections on "effect of friction angle on piling" and "comparison to real-world footage"; numbers were not retrieved — [paper, UCLA](https://www.math.ucla.edu/~cffjiang/research/sand/paper.pdf); [UC Davis copy](https://math.ucdavis.edu/~jteran/papers/KGPSJT16.pdf); [eScholarship dissertation](https://www.escholarship.org/content/qt5th3x8fb/qt5th3x8fb.pdf?t=o86y1e)
- **RFT (Li, Zhang & Goldman, Science 339:1408, 2013):** the force on an element is a stress per unit depth alpha_{z,x}(beta, gamma) × depth × area, where beta is the attack angle and gamma the intrusion angle. The generic fit is a Fourier series: alpha_z = Σ A_mn cos(2π(mβ/π + nγ/2π)) + B_mn sin(...), and alpha_x = Σ C_mn cos(...) + D_mn sin(...). Generic coefficients (N/cm^3, ζ = 1): A00 = 0.206, A10 = 0.169, B11 = 0.212, B01 = 0.358, B-11 = 0.055, C11 = -0.124, C01 = 0.253, C-11 = 0.007, D10 = 0.088 — [arXiv 1303.7065 (HTML)](https://arxiv.org/html/1303.7065)
- **RFT scaling factor ζ (with α ≈ ζ·α_generic):** poppy seeds loose 0.248, poppy seeds close-packed 0.488, 0.3 mm glass spheres loose 0.194 and close-packed 0.371, 3 mm glass spheres close-packed 0.214. For a new medium, measure α_z(0, π/2) with a horizontal plate pushed down and set ζ ≈ 0.8·α_z(0, π/2) — [arXiv 1303.7065](https://arxiv.org/html/1303.7065)
- **RFT measurement setup:** plate 3.81 × 2.54 cm (≈ 9.68 cm^2), 0.64 cm thick; depths 2.54, 5.08 and 7.62 cm; speed about 1 cm/s — [arXiv 1303.7065](https://arxiv.org/html/1303.7065)
- **RFT accuracy:** c-leg peak force within 10% in four media and within 33% for all legs and media. Robot speed within 20% for 90% of gaits and within 35% for all. The scaling routine gives about 20% error on granular media and about 35% on natural sands. Depth-only models overpredicted lift by up to 1400% — [arXiv 1303.7065](https://arxiv.org/html/1303.7065)
- Follow-up: Zhang & Goldman 2014, Phys. Fluids, "The effectiveness of resistive force theory in granular locomotion" — [Goldman lab PDF](https://goldmanlab.gatech.edu/pages/publications/pdf/ZhangGoldman_POF_2014_1.4898629corrected.pdf)

### Inferences
- **Recommended RFT scene:** a 3.81 × 2.54 × 0.64 cm rigid plate driven kinematically at 1 cm/s through a bed of loosely packed 0.3 mm-glass-like Drucker-Prager material. Sweep beta and gamma over a 7 × 7 grid and depths of 2.5-7.6 cm. Fit the linear-in-depth slope to get alpha_z and alpha_x, then compare (a) Newton vs Metal and (b) each vs ζ·generic. A tolerance of about 20% matches the paper's own model-vs-experiment accuracy. Agreement between the two simulators should be much tighter (< 5-10%) because the two continuum models are the same.
- **Silo scene:** a flat-bottom 2D or 3D hopper sweeping D/d_effective. Since MPM has no grain size, k·d is replaced by an empty-annulus scale set by grid spacing. Check that the W ∝ D^(5/2) exponent holds and compare the fitted C between codes.
- **Angle of repose:** pour or lift a cylinder at friction angles of 20, 30 and 40 deg. Report the pile slope against the input phi. Expect the slope to be below phi, per the Dunatunga & Kamrin observation.

### Gaps
- Bagnold-profile (inclined chute) numbers were not gathered.
- No published table of MPM repose angle against input friction angle was found.
- Bekker terramechanics parameters (k_c, k_phi, n) for standard sands were not retrieved.

## Fluids: dam break, hydrostatic, sloshing

### Takeaway
Martin & Moyce (1952) is the canonical dam-break dataset. Scales differ between papers, so the normalization must be fixed explicitly: Z = x/a against T = t·sqrt(g/a) or t·sqrt(2g/a). The Ritter shallow-water limit gives a front speed of 2 sqrt(gH). Experiments give 1.48-1.69 sqrt(gH).

### Cited Findings
- Martin & Moyce, Phil. Trans. R. Soc. A 244(882):325-334 (1952): collapse of rectangular and cylindrical water columns. It contains a "space-time relation for surge front" section and non-dimensional variables to compare scales — search summaries pointing to [IAHR library](https://www.iahr.org/library/info?pid=23741) and [Staroszczyk SPH paper](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-article-BAT8-0017-0012/c/Staroszczyk.pdf)
- Recreations normalize the front by a and time by sqrt(g/a). Lobovský et al. 2014 use t* = t·sqrt(g/H) — [arXiv 1308.0115](https://arxiv.org/pdf/1308.0115); [Leng & Chanson EFMC 2016](https://staff.civil.uq.edu.au/h.chanson/reprints/Leng_Chanson_efmc_2016.pdf)
- Martin & Moyce runs at H = 57 mm and 114 mm give a non-dimensional mean front velocity v/sqrt(gH) of 1.48 and 1.69. Both are below Ritter's analytic value of 2 — [arXiv 1308.0115](https://arxiv.org/pdf/1308.0115)
- The broken-dam problem is a standard low-Mach benchmark — [NTU benchmark PDF](https://www.math.ntu.edu.tw/~shyue/orch08/benchmarks/lowmach/broken-dam/91520.pdf)

### Inferences
- **Recommended scene:** a square water column (a = H, aspect ratio 1 or 2) in a closed tank. Plot Z(T) up to T ≈ 3, and also a hydrostatic column, checking p(z) = rho g (h - z) at rest. MPM with a weakly compressible EOS should give a front within about 5-10% of the experimental curve. Inter-simulator agreement should be tighter (inference).
- Sloshing (Isaac's Franka-Pour-style cup) is better compared at the statistical level, using fill fraction transferred.

### Gaps
- The Z-T data table from the 1952 paper was not retrieved. Sloshing benchmark numbers (e.g., SPHERIC Test 10) were not gathered.

## Elastic tests and PIC/APIC energy behaviour

### Takeaway
No fetched source supplied numerical tolerances. MLS-MPM (Hu et al. 2018) is the formulation most GPU codes (Taichi, PlasticineLab, ManiSkill2/Warp, Genesis) build on. It is equivalent to APIC for transfers and about 2× faster through its stress-divergence discretization.

### Cited Findings
- MLS-MPM: "all MPM simulations run two times faster than before" via a new stress-divergence discretization. It is consistent with APIC/PolyPIC under a Galerkin-style weak form — [Hu et al. 2018 project page](https://yuanming.taichi.graphics/publication/2018-mlsmpm/); [paper PDF](https://www.math.ucla.edu/~cffjiang/research/mlsmpm/hu2018mlsmpm.pdf)
- AsyncTaichi fuses G2P with the next P2G for extra speed in an MLS-MPM falling-cube benchmark — [arXiv 2012.08141](https://arxiv.org/pdf/2012.08141)

### Inferences
- **Recommended elastic scenes** (analytic ground truth, no source fetched; standard mechanics):
  - Cantilever tip deflection δ = PL^3/(3EI) under small load.
  - 1D bar P-wave speed c = sqrt(E/rho), or sqrt(E(1-ν)/((1+ν)(1-2ν)rho)) for the 3D bulk.
  - Ball drop with no plasticity, checking total energy drift over N steps.
  - Two-ring collision (Sulsky/Li tests).
  These need no experimental data and give exact numbers to compare both simulators against.

### Gaps
- No retrieved numbers for APIC vs PIC energy dissipation (Jiang et al. 2015, APIC paper) or cantilever errors for MPM.

## Robotics benchmarks with soft-body / granular tasks

### Takeaway
ManiSkill2's six MPM tasks have crisp, reproducible success criteria and published physical parameters, which makes them the best ready-made robotics comparison. Isaac Lab now ships a Newton-MPM Franka-Pour task. RFT-style foot penetration (above) is the best locomotion-relevant quantitative test.

### Cited Findings
- **ManiSkill2 soft-body simulator:** MLS-MPM implemented in NVIDIA Warp, with two-way rigid coupling via SDF penalty forces and several soft substeps per rigid step. Parameter ranges: grid spacing 0.005-0.015 m, particle volume 6.2e-8 to 1.2e-7 m^3, density 300-3000, E = 1e4-3e5 Pa, ν = 0.3, yield stress 2e3-1e4 — [ManiSkill2, arXiv 2302.04659](https://ar5iv.labs.arxiv.org/html/2302.04659)
- **ManiSkill2 success criteria:**
  - Fill: clay > 90% of target-beaker capacity, soft-body velocity < 0.05.
  - Hang: noodle above the rod, ends on opposite sides, not touching the ground, gripper open, v < 0.05.
  - Excavate: lifted amount within a range, above a height, < 20 particles spilled, v < 0.05.
  - Pour: liquid level within 4 mm of the line, < 100 particles spilled, bottle upright, arm v < 0.05.
  - Pinch: Chamfer distance to target < 0.3 × the initial Chamfer distance.
  - Write: IoU > 0.8 with the target character.
  — [arXiv 2302.04659](https://ar5iv.labs.arxiv.org/html/2302.04659); [arXiv 2410.05756](https://arxiv.org/pdf/2410.05756)
- **ManiSkill2 throughput:** about 17-18 FPS for one soft-body env with rendering, and 80-84 FPS total for 16 envs on one RTX Titan (about 4× real time). The limits come from sequential CPU-rigid/GPU-soft stepping and from CUDA kernel-launch overhead — [arXiv 2302.04659](https://ar5iv.labs.arxiv.org/html/2302.04659)
- **ManiSkill2 behaviour-cloning success (point cloud / RGBD):** Fill 0.45/0.62, Hang 0.35/0.20, Excavate 0.08/0.21, Pour 0.02/0.00, Pinch 0/0, Write 0/0 — [arXiv 2302.04659](https://ar5iv.labs.arxiv.org/html/2302.04659)
- **Isaac Lab + Newton implicit MPM:** the minimal example uses voxel size 0.02 m (solver grid equals the particle generator), particles_per_cell = 2, placement "cell_center", max_iterations = 100, tolerance = 1e-4, dt = 1/100 s and num_substeps = 2. Grid jitter uses a fixed seed. Damping is authored as damping × E. MPM and rigid-MPM coupling are marked experimental. The Franka-Pour task (`IsaacContrib-Franka-Pour`) uses source_fill_level 0.70 by default, a pour_target_frac success metric, a 20,000-row reset artifact and an example of 2048 envs. No performance figures are given — [Isaac Lab "Using Implicit MPM"](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/using_mpm.html)
- Newton ships `mpm_granular` and `mpm_anymal` examples (ANYmal on sand), plus two-way coupling, multi-material and snowball examples — [Newton on PyPI](https://pypi.org/project/newton/); [NVIDIA blog announcing Newton](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation)
- Genesis MPM docs cover configuration only, noting that domain bounds affect performance. They give no throughput numbers — [Genesis MPM options](https://genesis-world.readthedocs.io/en/v0.3.7/api_reference/options/simulator_coupler_and_solver_options/mpm_options.html)

### Inferences
- Using the Newton `mpm_anymal` and `mpm_granular` scenes is the most direct path to apples-to-apples comparison: same geometry, same voxel size (0.02 m) and same ppc (2), rerun in the Metal implementation. Compare foot sinkage depth per stance, base height and CoM velocity statistics, and particle-mass displacement.
- ManiSkill2-style scalar metrics (fill fraction, spilled particle count, Chamfer/IoU) are reproducible across simulators if the controller is replayed open-loop from logged actions. Closed-loop policies amplify small differences.

### Gaps
- No quantitative numbers retrieved for PlasticineLab, SoftGym, DexDeform, Bekker wheel sinkage, or published sim-to-real granular locomotion (e.g., Choi et al. 2023 "Learning quadrupedal locomotion on deformable terrain", Science Robotics; not fetched).

## Throughput metrics and published numbers

### Takeaway
Published GPU MPM numbers are scene-specific and rarely reported in particles·steps/s. Gao 2018: 4.2M particles on a 256^3 grid at 10.5 s per 48 Hz frame (implicit with heat, Quadro P6000). Fei et al. 2021: 1.33M particles in real time at 68.5 FPS on 4× V100. JAX-MPM: about 2.7M particles × 1000 steps in about 22 s on an A100, roughly 1.2e8 particle-steps/s (derived). No Apple Silicon MPM numbers were found.

### Cited Findings
- **Gao et al. 2018** (TOG/SIGGRAPH Asia, "GPU Optimization of Material Point Methods"): a 4.2M-particle elastoplastic dragon on a 256^3 grid (implicit MPM plus heat) at an average 10.5 s per 48 Hz frame on a Quadro P6000. Up to 10M particles in under 1 minute per frame — [search summary of GPU_MPM.pdf](https://people.csail.mit.edu/kuiwu/GPU_MPM/GPU_MPM.pdf) (per-kernel tables not extracted)
- **Fei, Huang & Gao 2021** ("Principles towards Real-Time Simulation of MPM on Modern GPUs"): 1.7-8.6× speedup on one GPU and 2.5-14.8× on four GPUs compared with the prior state of the art. Snow with 1.33M particles at 68.5 FPS and a fountain with 143K particles at 55.9 FPS, both on 4× Tesla V100 — [arXiv 2111.00699](https://arxiv.org/pdf/2111.00699)
- **JAX-MPM:** 1000 steps for over 2.7M particles on a 400×400×100 grid in about 22 s on an A100 (fp32) — [arXiv 2507.04192](https://arxiv.org/pdf/2507.04192)
- **CRESSim-MPM** benchmarks on an RTX 4090 across particle counts, but explicitly does not aim to beat existing codes on speed — [arXiv 2502.18437](https://arxiv.org/pdf/2502.18437)
- **HOT** (Wang et al. 2020): implicit MPM up to 10× faster than optimized implicit baselines — [HOT paper](https://www.math.ucla.edu/~cffjiang/research/hot/wang2020hot.pdf)

### Inferences
- Derived JAX-MPM throughput: 2.7e6 × 1000 / 22 s ≈ 1.2e8 particle-steps/s on an A100 (estimate).
- Recommended reporting for both simulators:
  - Particle-steps/s (and ms/step per 1M particles) at fixed ppc and grid size, broken into P2G, grid update, G2P and sort.
  - Env-steps/s for batched robot tasks.
  - Wall-clock real-time factor, for comparison with ManiSkill2's about 4× RT with 16 envs.

### Gaps
- No published Newton implicit MPM, Genesis MPM, or Taichi mpm128/mpm3d particles-per-second figures were found. No Apple Silicon/Metal MPM throughput numbers were found.

## Comparing two MPM codes: trajectory vs statistical level

### Takeaway
No fetched source prescribes a protocol. The literature practice seen here compares scalar end-state observables and their fitted scaling laws (runout, exponents, deposit profile, Beverloo C and exponent, RFT α curves), not particle trajectories. This suits chaotic granular flow and nondeterministic GPU atomics.

### Cited Findings
- Column-collapse numerics are judged by fitted power laws and deposit-shape similarity against experiment, not by trajectory error — [arXiv 1602.07881](https://arxiv.org/html/1602.07881v1)
- Isaac Lab MPM uses fixed-seed grid jitter shared across clones, so independent distributions need explicit randomization. Multi-seed statistics must therefore be set up deliberately — [Isaac Lab MPM docs](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/using_mpm.html)

### Inferences
- Proposed protocol:
  1. **Deterministic checks first.** Elastic and hydrostatic tests should match within float tolerance trends, e.g., less than 1% on δ, c and p(z).
  2. **Short-horizon trajectory checks.** Compare the CoM trajectory and KE(t) of the whole body over the first few hundred steps, before divergence.
  3. **Statistical checks.** For chaotic flows, compare across N ≥ 10 seeds (jittered particle positions): runout mean ± std, height-field h(x) RMS difference or Chamfer distance between deposits, KE(t) envelopes, and time-to-arrest. Pass if the two simulators' means differ by less than the combined 2σ and by less than about 10%.
  4. **Robot tasks.** Compare success rates and spill counts under replayed open-loop actions.

### Gaps
- No paper was found that formally benchmarks two MPM implementations against each other with a statistical protocol.
