"""CLI dispatch. Spec §3.3."""

import pytest

from src.cli import build_parser, main


def test_parser_accepts_all_spec_modes():
    p = build_parser()
    spec_modes = [
        "sandbox",
        "snapshot",
        "demo",
        "replay",
        "record",
        "eval",
        "openscene_static",
        "compare_backends",
    ]
    for mode in spec_modes:
        args = p.parse_args(["--mode", mode])
        assert args.mode == mode


def test_parser_rejects_unknown_mode():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--mode", "fly"])


def test_sandbox_mode_with_target_returns_zero():
    rc = main(["--mode", "sandbox", "--target", "0.5", "0.0", "1.2"])
    assert rc == 0


def test_unimplemented_mode_returns_nonzero():
    rc = main(["--mode", "record"])
    assert rc == 3


def test_snapshot_without_image_returns_2():
    rc = main(["--mode", "snapshot"])
    assert rc == 2
