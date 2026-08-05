# Structural learner-world specification

## Shared observation mapping

For skill `k`, item bin `b`, and normalized latent quantity `Q`, all three base worlds use the same response mapping

`P(Y=1 | Q,k,b) = g_k + (1-s_k-g_k) Q + item_shift(k,b)`

with the implementation applying the documented probability bounds. This common mapping is a benchmark construction for comparability, not an empirically established measurement-invariance result.

## Binary BKT-F benchmark

The binary state is mastered or not mastered. Initial mastery, learning, slip, guess, and forgetting are derived from precursor EdNet BKT-F fits and then empirical-Bayes regularized. Item-bin shifts are anchored to marginal historical item accuracy. Learner-level forgetting heterogeneity uses the benchmark lognormal multiplier. The complete derived world is not independently held out end to end.

## Continuous latent-state construction

The state is continuous on `[0,1]`. Learning follows `Q' = Q + (1-Q)p_L` and forgetting is implemented through the shared exponential retention structure. Its initial distribution is chosen to match the regularized initial mean, with `sigma_theta = 0.70` in the base specification. This is a designed structural stress construction.

## Four-state semi-Markov construction

The state occupies four ordered levels with normalized factors used by the shared response mapping. Progression depends on state and opportunity age, including the opportunity-age multiplier implemented in the simulator. Parameters are chosen to match selected first-order targets of the binary benchmark. This is a designed structural stress construction.

## Challenge-dependent learning family

The challenge family changes the learning treatment effect as a function of the policy-side expected success probability. The effective learning power is maximal near a target success probability and declines according to a width and floor parameter. A skill-specific normalization constant preserves the expected first-opportunity gain under a declared reference action distribution. Both uniform-bin and historical practice-bin-frequency normalizations are implemented.

The challenge family is a counterfactual structural sensitivity, not an empirically estimated learning law.

## Timing

The default practice schedule has 100 decisions, five items per simulated day, and five minutes between within-day decisions. All skills are synchronized to the common practice-end time before immediate evaluation. Delayed evaluation is then computed after the specified retention interval. Separate schedule sensitivity varies practice budget and items per simulated day.
