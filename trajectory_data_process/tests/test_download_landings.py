"""The legacy-named CLI is only a multi-airport adapter for harvest."""

from trajectory_data_process.download_landings import build_parser, harvest_argv


def test_wrapper_forwards_one_airport_to_the_canonical_harvest_cli():
    args = build_parser().parse_args(
        [
            "--airports", "krdu",
            "--count", "12",
            "--entry-radius-km", "20",
            "--evaluate-only",
            "--no-publish",
            "--multiplier", "30",
        ]
    )

    argv = harvest_argv(args, "KRDU")
    assert argv[:4] == ["--airport", "KRDU", "--count", "12"]
    assert argv[argv.index("--entry-radius-km") + 1] == "20.0"
    assert "--evaluate-only" in argv
    assert "--no-publish" in argv
    assert argv[argv.index("--multiplier") + 1] == "30"
