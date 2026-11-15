MATH 463 miniproject code (Hindmarsh-Rose bursting model + SINDy)
Requires: numpy, scipy, matplotlib, pysindy (2.1.0).
hr_model.py               HR model, LSODA simulation, spike/burst measurements
part1_replicate.py        Questions 2-3: replication figure, varying b and epsilon
sindy_experiments_lib.py  SINDy fitting and scoring helpers
sindy_experiments.py      Question 4 experiments (writes results.json, ~7 min)
make_figures.py           Question 4 figures from results.json
results.json              saved output of sindy_experiments.py
Run order: python part1_replicate.py; python sindy_experiments.py; python make_figures.py
