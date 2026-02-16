# Capture-mark-recapture models in Stan

This repo contains Stan programs for fitting a repertoire of capture-mark-recapture (CMR) models in Stan, specifically (Cormack-)Jolly-Seber models with and without robust design sampling in both single- and multistate configurations, including multievent models. Example Stan programs of each configuration are found in the `stan` folder, with the log likelihood functions found in accompanying `stanfunctions` files. All functions are overloaded to accommodate either (1) parameters varying by survey (or secondary) only or (2) parameters varying by individual. If no individual effects are required, the former is considerably faster, especially for Jolly-Seber models where the log likelihood of an augmented individual only has to be computed once. Jolly-Seber models feature additional "collapsed" function signatures, i.e. `js*2` and `*2_rng()`, that accommodate individual effects for observed individuals but only one log likelihood computation for augmented individuals. All functions were written as efficiently as possible to allow the practitioner to focus on flexibly modeling the model parameters without having to adjust the individual log likelihood computations.

All models feature the following by default:

1.  Mortality hazard rates and transition rates instead of probabilities to accommodate unequal survey intervals;

2.  Time-varying entry probabilities with an offset for survey length to accommodate unequal survey intervals;

3.  Individual log likelihoods stored in the `log_lik` variable to accommodate PSIS-LOO with the `loo` package, and the prior log density stored in the `lprior` variable to accommodate prior sensitivity analysis with the `priorsense` package.

Additionally, all Jolly-Seber models feature `_rng` Stan functions in `js-rng.stanfunctions` that return population sizes ($\boldsymbol{N}$), number of entries ($\boldsymbol{B}$), and number of exits ($\boldsymbol{D}$) per survey, as well as the super-population size. In multistate Jolly-Seber models, these quantities are returned by each state.

All configurations were tested with simulation-based calibration (SBC), available in the `sbc` folder.

The `examples` folder contains code and Stan programs to run the examples in the manuscript entitled "An overview of capture–mark–recapture models with efficient Bayesian implementations in Stan".
