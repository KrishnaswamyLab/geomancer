"""Expert-guided training mode: run the CFlows workflow one step at a time, keep every
iteration, choose what goes forward, and record why.

    choices.py   every value the lab reviews
    workflow.py  the steps and their parameters (pure data)
    session.py   iterations, selection, staleness, persistence, trace (no compute)
    execute.py   the one module that computes, through manyruns
    timebins.py  refined time bins around real timepoints
    diagnostics.py  cheap per-step numbers beside each plot, read by the rules
    feedback.py  plain-language feedback → proposals (rules, optional model); no side effects
    loop.py      the interactive loop and the one-shot feedback/compare verbs
    cli.py       `geomancer train …`
"""
