"""A minimal learner over scShapeBench: pick an action from cheap input measurements.

An action is an embedding, then a graph extractor, at one scale. `table` runs every
action on every sample once and writes the rewards; policies replay that table.
Everything is a plain function or dict. The ablation axes are the dicts
`state.STATES`, `actions.ACTIONS` and the reward functions in `rewards`.
"""
