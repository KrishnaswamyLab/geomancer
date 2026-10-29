"""The scShapeBench action grid's embedding recipes."""
from pathlib import Path

import pytest
import yaml

RECIPES = Path(__file__).resolve().parents[1] / "geomancer" / "configs" / "recipe"


@pytest.mark.parametrize("name,step", [("ssb_embed_pca", "pca"),
                                       ("ssb_embed_diffmap", "diffusion_map"),
                                       ("ssb_embed_phate", "phate")])
def test_embedding_recipes_are_one_latent_step_at_manylatents_defaults(name, step):
    from manyruns import catalog

    recipe = yaml.safe_load((RECIPES / f"{name}.yaml").read_text())

    assert catalog.check_recipe(recipe) == []
    assert [s["name"] for s in recipe["steps"]] == [step]
    only = recipe["steps"][0]
    # Defaults, except no landmarks for diffusion maps: at the class default (2000) every sample
    # fails (decided 2026-10-29; the recipe's header records both failure modes).
    expected = {"n_components": 2, **({"n_landmark": None} if step == "diffusion_map" else {})}
    assert only["group"] == "latent" and only["params"] == expected
    assert only["limits"] == {"n_components": ["n_samples", "n_features"]}
