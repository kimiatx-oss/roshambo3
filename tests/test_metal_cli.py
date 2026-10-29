from types import SimpleNamespace

import pytest

from roshambo3 import cli
from roshambo3.backend.spec import ColorPlacement, PoseSearch


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
def test_metal_cli_modes(monkeypatch, mode, capsys):
    import roshambo3

    observed = {}

    def screen(*a, **kw):
        observed.update(kw)

        class Result:
            stats = SimpleNamespace(n_conformers=0, wall_s=0.1, conformers_per_s=0)

            def __len__(self):
                return 0

            def __iter__(self):
                return iter(())

        return Result()

    monkeypatch.setattr(roshambo3, "screen", screen)
    assert (
        cli.main(
            [
                "screen",
                "q.sdf",
                "corpus",
                "--backend",
                "metal",
                "--mode",
                mode,
                "--level",
                "molecule",
            ]
        )
        == 0
    )
    assert observed["backend"] == "metal"
    cfg = observed["config"]
    assert cfg.pose_search == PoseSearch.FIELD
    assert cfg.color == (mode != "shape")
    assert cfg.mixing == (0.5 if mode != "shape" else 0.0)
    assert cfg.color_placement == (
        ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE
    )
