"""Expert-guided training mode: run the CFlows workflow one step at a time, keep every
iteration, choose what goes forward, and record why.

    choices.py   every value the lab reviews
    workflow.py  the steps and their parameters (pure data)
    session.py   iterations, selection, staleness, persistence, trace (no compute)
    execute.py   the one module that computes, through manyruns
    timebins.py  refined time bins around real timepoints
    cli.py       `geomancer train …`
"""
