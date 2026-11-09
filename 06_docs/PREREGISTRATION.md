# Preregistration draft (post on OSF before any adjusted model on real data)

Disclosure: during planning, unadjusted Sim 4 reporting rates by Sim 3 status were seen.
No adjusted model has been run on the real data.

Data: open data of George et al. (Zenodo v4, md5 3cb0e6f97e5cfac0906e0c3cd29029d5), corrected
attention rule (N = 1,269) for effects, published rule (N = 986) for the benchmark. KNUST survey (target 250).

Hypotheses:
- H1: being caught in Simulation 3 raises the probability of clicking in Simulation 4 (ATE > 0).
- H2: being caught in Simulation 3 lowers the probability of reporting in Simulation 4 (ATE < 0).
- H3: the click effect is larger for staff who clicked in Simulations 1-2.
- H4: among Simulation 3 clickers, seeing the training page lowers Simulation 4 clicking.
- H5: five scales reach metric and at least partial scalar invariance between the two universities.

Analysis: DoubleML IRM, 5 folds x 10 repeats, propensity clipped at .02, logit (primary) and
gradient boosting (check). Covariates: settings.COVARIATES only. GATES for role, age band, prior
clicking, with an F test of equal group effects. Sensitivity bounds (cf_y = cf_d = .03), E-value,
negative-control outcome (Sim 2 click). Invariance: lavaan MLR, delta CFI <= .01 and delta RMSEA <= .015.
Alpha .05, two-sided. Group cells with fewer than 10 treated staff are merged before reporting.
