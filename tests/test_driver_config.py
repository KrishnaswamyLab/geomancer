"""The lab's driver config: a knob card for `manyagents-drive --knobs`, and the models it names."""
import json
from pathlib import Path

import yaml

DRIVER = Path(__file__).resolve().parents[1] / "geomancer" / "configs" / "driver"


def test_the_knob_card_explains_only_knobs_manyruns_offers():
    from manyruns.tui.params import TUNABLE

    assert DRIVER.is_dir()
    card = json.loads((DRIVER / "knobs.json").read_text())
    offered = {name for names in TUNABLE.values() for name in names}
    assert card and set(card) <= offered, sorted(set(card) - offered)
    assert all(isinstance(v, str) and v.strip() for v in card.values())


def test_every_model_entry_says_who_it_is_and_how_to_reach_it():
    assert DRIVER.is_dir()
    models = yaml.safe_load((DRIVER / "models.yaml").read_text())
    for entry in models["baselines"]:
        assert entry["chooser"].startswith("llm:") and entry["agent"] and entry["model"]
