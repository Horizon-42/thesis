"""A runway without LPV can still publish its vertical path: its RNAV (GPS) approach's
runway leg. These pin the decode against the Path Points, the two fleet runways it
recovers (KRDU 32, KSMF 35R), the one it cannot (KRDU 14 has no procedure), and --
because experiments run against the stored events -- that no LPV runway's
fingerprints moved."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.context import NO_VERTICAL_GUIDANCE_SOURCE, assessment_for_runway
from trajectory_data_process.harvest.airports import (
    APPROACH_LEG_TCH_SOURCE,
    PATH_POINT_TCH_SOURCE,
    load_airport,
    runway_data_fingerprint,
    threshold_frame_fingerprint,
)
from trajectory_data_process.harvest.cifp import (
    FT_M,
    ApproachVertical,
    PathPoint,
    _verify_approach_decode,
    read_approach_verticals,
    read_path_points,
)

CIFP = Path("data/CIFP/CIFP_260806/FAACIFP18")
CONFIG = Path("trajectory_data_process/config/runway_thresholds.json")
FLEET = ("KRDU", "KSJC", "KSTL", "KSMF", "KMSY")

# (frame fingerprint, evaluation-context fingerprint) of every runway at commit 3029b09,
# before the approach-leg vertical path existed. ``threshold_frame_snapshot`` never
# hashed the TCH, so all 26 frames must survive (every stored threshold event stays
# valid); the context does hash it, so only the 23 LPV contexts may. The third digest,
# ``runway_data_fingerprint`` (``_BASE_RUNWAY_DATA``), hashes every Runway field and is
# EXPECTED to move on all 26 -- it is provenance written into reclassify/rebuild
# manifests and checked by nothing in production.
_BASE = {
    "KRDU 05L": ("f886dcc9611f1a9cef56a43da7fc9671f00f8e5b4db78740cb30b97f3c1c11e0", "b759142e996a6066cc9b8ebcc7112cd6ec9ae0f82f251869e2c8d9d32719727e"),
    "KRDU 23R": ("b49649c85c0805cb8a3ca4eca7cdf6e86dcd2274d6d3e4bf6c60afe8ea20dac5", "c133e608ad94fffb5289c2120d4460ffeed12f9781efd5693a664e2ab2e41493"),
    "KRDU 05R": ("58b8e65a6c5321fe113319e844069bb001602d272a08664c2f1cb04777df576e", "6e3bca1aa6faedd9054fbb2fc3a713a4d8cc2a4456434fc30be8bfe424b1c226"),
    "KRDU 23L": ("0713bc8262e9752c40da147cea9a27b3dcefa8f81b1161260e30ac769c21c566", "8424eb4c1ee860ba7e8c62b53749ee89638499beca414b9854a5d368b58fe4fb"),
    "KRDU 14": ("6a44a065869f2c70890a35cf25d2fdf70ab74b79a398d2e33b25a9ecfcfeaa29", "b0fc71613a919b09b5c8e07db7213f24ce341c0c0d3166af6d9291bb766e564c"),
    "KRDU 32": ("2fa4523787f71511b301161b582cee41d8d06d29a6f4c01e4cc46ce0b379e52d", "adc618473c914d028c7b7d0271dfbdf5e8d981c6b9d0d83dae80af6630c5b9e6"),
    "KSJC 12L": ("fef40fe5927ffc9698b207baefc891c0525268ee6eee7e30ecc544f549ce6322", "49d1766a457059369c91cc95eab88747c6cc92bcb219b2e72f71715e595f3f06"),
    "KSJC 30R": ("239c1fd463d655e56db97e61e6ebaf90b2ca1bedd60d72efb6c9048815149df0", "f6b50f3cc691632f0425e40d4192dbff6d378a1364a8a720dc2125cab9a0f579"),
    "KSJC 12R": ("bf244e8feb209f782d92379ceaba4dc954312d30697c0f3487da406fd92bccb4", "e15ff049110c49651554571c6301f8757f3933fdbf58a3a71d747ccc31810581"),
    "KSJC 30L": ("375cbf592331ee61de92e1f86ad8f26b827a6b97b7d2854c1e5540639770a057", "6d87853da4d505a45d7f9150c4ec2e433ea898dab219bc7cedbcd76315356e39"),
    "KSTL 06": ("c59b7c0c7395bb2b179b0eb12ef593ef8ef4ed0be8e7726d6efd64c2c6167b7b", "3381c3274c305f8614f2d8e4601dc0e4213751fc17a265d9a4d425bc854d4ccb"),
    "KSTL 24": ("4a2f55877f740e97610caaa72e48813ff3a2cac7bd2ad928055e1942d0d91d98", "666d18b369ad0a6c6bceabb0ca3c805475e02e5f630a09e056793e7fa7d8cc03"),
    "KSTL 11": ("b3b878f09aa7f599ab1d7d23278dcfdf1f6cc5a596d955ee65354855622f0597", "3d224e6f409fb93c4dce4d021fb0fcc04c99ed43af6d4616afec4a3ca858459d"),
    "KSTL 29": ("f864d44b3ff253626dbd7ba690660e9c00e0587b7447a1b3dc86523494ea676c", "997d4816d4ecb1678e45549263ad1d664f871c88a890d90dda8549da4d7a10ba"),
    "KSTL 12L": ("5e14c86667d11b9cf7e720783d84d002fab94cc596d784f78c9a928055ce98c6", "8fee0268ad27772da946703edd03d0ec2bd8e1b01fa63a6cae5a343aff7b2e99"),
    "KSTL 30R": ("a0e1d0a2e7a066cbf850bc09a52a8459278dc58d25b2b6fd581cdda94bf81c14", "a9497ca1ea92dd8b3fc8aa47c86ddbc94007e2507988420dfd8ea8608019bfda"),
    "KSTL 12R": ("447a676e46551ccb3a2cfbd75b9493f79d02916f2d1f72d80334ef75dfbc136b", "1e4f835b3f2ea3c54cd9ec3ad4b76b6a86ebd84cc1a387ccd84775667d996728"),
    "KSTL 30L": ("da60d071a398eeca2222ff6c4b24eac6795969a8c07c03158c99e3bf5fb4ba15", "660e9a12e6f0abddf403e138f5555a4321f8eecfb42d4fca59d006a58d2a4cbe"),
    "KSMF 17L": ("322f9543e0cfcee144efee4edc67ce3c878fee3d25c1667995edc5b8a48fa4f7", "84f6475ea58d9267fe0068a43175223aa174f21197a1c1441173ecec54e730d1"),
    "KSMF 35R": ("bebc7e0d6ae5c34afbb9cdbdbfcfcada324b80cdb91b07eacc2fdab248090c4b", "dc5ca8f07dcadad4e9c56c16882cbe162652e8d4f5593849f9a5ae8d900973b8"),
    "KSMF 17R": ("4035c333f1568947be800945900f4bb868473e9921887c0da35c81cfe7035ce1", "eba85fdddf6fcebb459a4272b402c6bc28bf6ece5b175623d2a1ad4233ffd3c6"),
    "KSMF 35L": ("e3186a4221a60beb1cf57e350f738f95c4d3854cfc86ca7cdaa88f0b33933b52", "34b527c53f0b27357d6f2b2d644b94468118b914df9e51b0fffe7b58be70c329"),
    "KMSY 02": ("e9d0e666b905ed2c6acb4f9c26ac7b1916230f98dda90e73c7a822d6c4c2ea7e", "e6bfc11ea3b6062ac10e8e8bb20c1056af2d91bdaac3b438901a9fd73a49965b"),
    "KMSY 20": ("b000e5ebb02692ab7cee016feefe6e507b1de96184f843f0cc59605f07e1144c", "7a193eeb99e5d8795a69f8b0951f225311eed1e355cf169b8504fd5a2ed1f952"),
    "KMSY 11": ("eb828f8eb408d6593c46ef0d133cc94f1d558c512b7626400556928a1910b796", "88d7879dfe4801c5c767fe9d7571bc6390fe202d406a78fefca7632657e689b5"),
    "KMSY 29": ("fcbf0f65a377a0a47776b3d29c10227dc044fa411c3b41163ced7af9e6f0a966", "6490b668a7474b1ab322df5f9c1dbbf1360ad7c4df898bd51065aae93363fc08"),
}


_BASE_RUNWAY_DATA = {
    "KRDU 05L": "5219fbfde18afc5339901ca634ac27eea374825dd3f6ebd70e1f3375ddc4670b",
    "KRDU 23R": "40894334cffa226cacf8c2aa3ad7962bcd66f4231a836d79190bd7abcdbb9db7",
    "KRDU 05R": "c341617aba970e6b365e05429931396a592a440c9c1c853f3ed3710e510cf217",
    "KRDU 23L": "edbecba3eaaee1c2053c41914bc33a3881b2b86b71dc77b3d5a054b02e99e36d",
    "KRDU 14": "c39b4465c7880444107e9694ecd63a94245b28787c7699167f96116141ac58b9",
    "KRDU 32": "b95b0913f498bc4ca58e1dcc9fb2dad42c34f3c653939241d4ce59b348882143",
    "KSJC 12L": "6556531744f65c7a9f27f4aa662efb536cb4786e6fcad04d8fa46cf7ed67d2fe",
    "KSJC 30R": "5fd8e4fc22157ddded009424c1a7953ace09cdb1663fc5e1c3ddf12103474e2e",
    "KSJC 12R": "54dacce3fd40b46817e0685148bb68cb97b360630c418f06a134e60e0bfe52b0",
    "KSJC 30L": "90f2264768a9d58eccf4f6540c4696341fdecea2ec9cee02b45077a5bff484fb",
    "KSTL 06": "75309c79d6c36fe7335e85337b4a33363b34c515346b6fbc0e263b60501bf1df",
    "KSTL 24": "fd709010974cbdc9f611030397c3ac6f18ce8e94a96853bd88c7372070b24c9a",
    "KSTL 11": "d0199ccb0783c47ce9fe53f5ddcb02ed821ee47308137515cd6f1d3399aa0925",
    "KSTL 29": "7a6e54733fd4f7486a6f3f368413ef5debf3bbe70a87fd4e3b5c445da2b17ecd",
    "KSTL 12L": "31a957f3b260fac529ef946cd4079285d9ed610c3adfb43c86f83bd8c4da68f6",
    "KSTL 30R": "d5447d34725342e9fefdd634999a2ca113b2e2ac6c18cb0a390b300c1f4e8314",
    "KSTL 12R": "8794d98bce56abb6bfa29642fbea58c95195958941acd48a4989c83441fbeae3",
    "KSTL 30L": "bdca5182a2cabb7e96f1fa52ae279099edf42a81c2a9ba34ebeada30abdbf15f",
    "KSMF 17L": "a4e14e6ea4443ac0d588a8a793cd5cf7878ab1b3e11a070e5a624db3e96ae458",
    "KSMF 35R": "40a90fa4e424225035dfdadb6ba3ac41312904f751f24e2bde47f50c8fa18b2b",
    "KSMF 17R": "a6f423f3480deed130f2b3a013e79e02166128d6fb6ddc63599dd5ccc143e862",
    "KSMF 35L": "b7a7b93c4d9acf307af0aa7a36ed611f9e90ddb7cd75d138f92735412373b97b",
    "KMSY 02": "de5226ef4896a88e8f7a9aa7e4e5b1c1f566662882eacd633911b6e04143b98f",
    "KMSY 20": "c7fc35e1bee0c49612d8b023374c25965bbe58cc106dedaf66a4e74c6fced7d1",
    "KMSY 11": "937c3712941dbe751da242a7e48e46db187f0ef30c75a3ea6fbdcd405dec2660",
    "KMSY 29": "51f12dd1ced5776f7332c1269d62549fa07209083039687dc5fcfb28862e1276",
}


def test_rnav_gps_runway_legs_publish_the_lnav_vnav_path():
    verticals = read_approach_verticals(
        CIFP, airport="KRDU", path_points=read_path_points(CIFP, airport="KRDU")
    )
    rw32 = verticals[("KRDU", "32")]
    assert rw32.procedure == "R32"
    assert rw32.glidepath_deg == 3.5
    assert rw32.crossing_altitude_msl_m == pytest.approx(470.0 * FT_M)
    assert rw32.baro_vnav_minima and not rw32.lpv_minima
    assert ("KRDU", "14") not in verticals            # no RNAV procedure at all
    assert verticals[("KRDU", "05L")].lpv_minima      # the LPV runways carry both lines


def test_load_airport_fills_the_vertical_path_from_the_approach_leg():
    krdu = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP)
    rw32 = krdu.runway("32")
    assert rw32.lpv_course_width_m is None
    assert rw32.tch_source == APPROACH_LEG_TCH_SOURCE and rw32.baro_vnav_minima
    assert rw32.published_glidepath_deg == 3.5
    # The leg crosses at 470 ft over a 425 ft threshold: the 45 ft the runway record
    # publishes too, but taken from the procedure that flies it.
    assert rw32.threshold_crossing_height_m == pytest.approx(45.0 * FT_M, abs=0.1)
    rw14 = krdu.runway("14")
    assert rw14.threshold_crossing_height_m is None
    assert rw14.tch_source is None and not rw14.baro_vnav_minima
    assert krdu.runway("05L").tch_source == PATH_POINT_TCH_SOURCE

    rw35r = load_airport("KSMF", config_file=CONFIG, cifp_file=CIFP).runway("35R")
    assert rw35r.published_glidepath_deg == 3.0
    assert rw35r.threshold_crossing_height_m == pytest.approx(64.0 * FT_M, abs=0.1)
    assert rw35r.tch_source == APPROACH_LEG_TCH_SOURCE


def test_lnav_vnav_runway_resolves_a_real_vertical_gate():
    krdu = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP)
    context = assessment_for_runway(krdu.runway("32"))
    assert context.benchmark == "rnp_apch_lnav_vnav_baro"
    assert context.baro_vnav_approved
    assert context.procedure_source == APPROACH_LEG_TCH_SOURCE
    limits = context.limits()
    assert (limits.vertical_lower_m, limits.vertical_upper_m) == (-22.0, 22.0)
    assert limits.vertical_reason is None
    assert context.desired_threshold_altitude_msl_m == pytest.approx(470.0 * FT_M, abs=0.1)

    no_procedure = assessment_for_runway(krdu.runway("14"))
    assert not no_procedure.baro_vnav_approved
    assert no_procedure.procedure_source == NO_VERTICAL_GUIDANCE_SOURCE
    assert no_procedure.limits().vertical_lower_m is None


def test_lpv_runway_fingerprints_did_not_move():
    seen = 0
    for code in FLEET:
        for runway in load_airport(code, config_file=CONFIG, cifp_file=CIFP).runways:
            frame, context = _BASE[f"{code} {runway.ident}"]
            seen += 1
            assert threshold_frame_fingerprint(runway) == frame, (code, runway.ident)
            actual = assessment_for_runway(runway).evaluation_context_fingerprint
            if runway.lpv_course_width_m is not None:
                assert actual == context, (code, runway.ident)
            else:
                # Gained a vertical path (32, 35R) or a named procedure source (14).
                assert actual != context, (code, runway.ident)
            # The provenance digest hashes the two new fields: moved everywhere, by design.
            assert runway_data_fingerprint(runway) != _BASE_RUNWAY_DATA[f"{code} {runway.ident}"]
    assert seen == len(_BASE) == 26


def _path_point(runway: str = "09", *, tch_m: float = 16.0) -> PathPoint:
    return PathPoint(
        airport="KAAA", runway=runway, latitude=0.0, longitude=0.0, glidepath_deg=3.0,
        threshold_crossing_height_m=tch_m, course_width_m=106.75,
        ltp_ellipsoidal_height_m=100.0, ltp_orthometric_height_m=130.0,
    )


def _vertical(runway: str, altitude_m: float, *, angle: float = 3.0, lpv: bool = True,
              conflict: str | None = None) -> ApproachVertical:
    return ApproachVertical("KAAA", runway, f"R{runway}", angle, altitude_m, lpv, True,
                            conflict=conflict)


def test_the_pin_is_a_confidence_rule_over_the_lpv_runways():
    """Three of four agreeing passes (a lone data disagreement does not condemn the
    airport); two of four, a wrong angle everywhere, or an unmatched LPV procedure
    raises -- the shapes a shifted column or a mis-keyed decode produce."""
    points = {("KAAA", rw): _path_point(rw) for rw in ("09", "27", "18", "36")}
    good = {("KAAA", rw): _vertical(rw, 146.0) for rw in ("09", "27", "18")}
    _verify_approach_decode(Path("x"), {**good, ("KAAA", "36"): _vertical("36", 150.0)}, points)
    with pytest.raises(ValueError, match="agrees with the Path Points on only 2/4"):
        _verify_approach_decode(
            Path("x"),
            {**good, ("KAAA", "18"): _vertical("18", 150.0), ("KAAA", "36"): _vertical("36", 150.0)},
            points,
        )
    with pytest.raises(ValueError, match="agrees with the Path Points on only 0/4"):
        _verify_approach_decode(
            Path("x"), {key: _vertical(key[1], 146.0, angle=3.1) for key in points}, points
        )
    with pytest.raises(ValueError, match="unverified"):
        _verify_approach_decode(Path("x"), {("KAAA", "05"): _vertical("05", 146.0)}, points)
    # A conflict is not a comparison; an airport with no LPV-bearing procedure has
    # nothing to pin against and is not an error.
    _verify_approach_decode(
        Path("x"),
        {**good, ("KAAA", "36"): _vertical("36", 999.0, conflict="Y and Z disagree")},
        points,
    )
    _verify_approach_decode(
        Path("x"), {("KAAA", "09"): _vertical("09", 146.0, lpv=False)}, points
    )


def _record(
    procedure: str,
    *,
    fix: str = "RW09 ",
    fix_section: str = "PG",
    transition: str = "     ",
    continuation: str = "0",
    altitude_ft: int | None = 146,
    altitude_description: str = " ",
    angle: str = "-300",
    application: str = " ",
    minima: str = "",
) -> str:
    """One 132-column procedure record: section P / subsection F for airport KAAA."""
    row = [" "] * 132
    row[4] = "P"
    row[6:10] = "KAAA"
    row[12] = "F"
    row[13:19] = f"{procedure:<6}"
    row[20:25] = transition
    row[29:34] = fix
    row[36:38] = fix_section
    row[38] = continuation
    row[39] = application
    if minima:
        row[40:40 + len(minima)] = minima
    row[82] = altitude_description
    if altitude_ft is not None:
        row[84:89] = f"{altitude_ft:05d}"
    row[102:106] = angle
    return "".join(row)


def _leg(procedure: str, altitude_ft: int, *, angle: str = "-300") -> str:
    return _record(procedure, altitude_ft=altitude_ft, angle=angle)


def _write(tmp_path: Path, *lines: str) -> Path:
    cifp = tmp_path / "FAACIFP18"
    cifp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cifp


def test_two_rnav_procedures_to_one_runway_that_disagree_are_a_conflict_not_a_choice(tmp_path):
    cifp = _write(tmp_path, _leg("R09Y", 146), _leg("R09Z", 150))
    [vertical] = read_approach_verticals(cifp, airport="KAAA").values()
    assert vertical.conflict and "publish different vertical paths" in vertical.conflict
    assert vertical.glidepath_deg is None and vertical.crossing_altitude_msl_m is None

    cifp = _write(tmp_path, _leg("R09Y", 146), _leg("R09Z", 146))
    [vertical] = read_approach_verticals(cifp, airport="KAAA").values()
    assert vertical.conflict is None
    assert vertical.procedure == "R09Y/R09Z"
    assert vertical.glidepath_deg == 3.0
    assert vertical.crossing_altitude_msl_m == pytest.approx(146 * FT_M)
    assert not vertical.baro_vnav_minima     # no approach-types continuation was coded


def test_only_the_final_segment_runway_leg_of_a_runway_procedure_is_read(tmp_path):
    """Transition legs, non-runway fixes, circling RNAV (GPS)-A and RNP AR procedures
    are not the crossing; a PRIMARY record whose waypoint-description letter sits in
    the continuation's application column is the leg, not an approach-types record;
    the approach-types continuation sets the minima flags."""
    cifp = _write(
        tmp_path,
        _record("R09", transition="ABCDE", altitude_ft=900),       # transition leg to RW09
        _record("R09", fix="FAFIX", fix_section="PC", altitude_ft=1400),
        _record("R09", application="G", altitude_ft=146),           # the runway leg
        _record("R09", continuation="2", application="W",
                minima="ALPV       ALNAV/VNAV ALNAV      "),
        _record("RNV-A", altitude_ft=900),                          # circling
        _record("H09", altitude_ft=140),                            # RNP AR
    )
    verticals = read_approach_verticals(cifp, airport="KAAA")
    assert list(verticals) == [("KAAA", "09")]
    vertical = verticals[("KAAA", "09")]
    assert vertical.procedure == "R09"
    assert vertical.crossing_altitude_msl_m == pytest.approx(146 * FT_M)
    assert vertical.lpv_minima and vertical.baro_vnav_minima


def test_a_procedure_that_codes_the_runway_fix_twice_or_a_bounded_altitude_raises(tmp_path):
    with pytest.raises(ValueError, match="codes the runway fix twice"):
        read_approach_verticals(
            _write(tmp_path, _leg("R09", 146), _leg("R09", 140)), airport="KAAA"
        )
    with pytest.raises(ValueError, match="bounded altitude"):
        read_approach_verticals(
            _write(tmp_path, _record("R09", altitude_description="+")), airport="KAAA"
        )
