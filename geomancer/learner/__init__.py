"""A minimal learner over scShapeBench: pick an action from cheap input measurements.

An action is an embedding, then a graph extractor, at one scale. `drive` runs every
action on every sample once through manyruns, one run record per cell; `score`
scores the graphs after the runs; `table` joins records and scores into the reward
table; `train` replays that table.
Everything is a plain function or dict. The ablation axes are the dicts
`state.STATES`, `actions.ACTIONS` and the reward functions in `rewards`.
"""
