"""六个核心缺陷修复的行为回归测试。

这些测试补充仓库自带的非空契约测试，重点验证题目描述中的：
实际间距标定、当前采样率折算、块间历史、轨迹平滑、极性纠正和温度补偿。
"""

import math

import pytest

from media import mic_array_beamform as mb


def approx(value, **kwargs):
    return pytest.approx(value, **kwargs)


def test_calibration_uses_actual_spacing_without_reversal():
    cal = mb.array_calibration([0.03, 0.05], sample_rate=48_000)

    assert cal["n_mics"] == 3
    assert cal["positions"] == [(0.0,), (0.03,), (0.08,)]
    assert cal["spacings"][0] == 0.03
    assert cal["spacings"][1] == 0.05
    assert cal["distance_matrix"][0][1] == 0.03
    assert cal["distance_matrix"][1][2] == 0.05
    assert cal["distance_matrix"][0][2] == 0.08
    assert cal["sample_delay_matrix"][0][1] == (
        0.03 / mb.SPEED_OF_SOUND * 48_000
    )


def test_delay_weights_follow_current_sample_rate():
    delay_48k = mb.delay_weight(0.001, 48_000)
    delay_44_1k = mb.delay_weight(0.001, 44_100)
    delay_ms = mb.delay_weight(1.0, 48_000, unit="ms")

    assert delay_48k == 48.0
    assert delay_44_1k == approx(44.1)
    assert delay_44_1k.integer == 44
    assert delay_44_1k.fraction == approx(0.1, abs=1e-10)
    assert delay_ms == 48.0
    assert mb.sample_rate(44_100) == 44_100


def test_block_history_matches_one_shot_delay_line():
    samples = [float(i % 7) for i in range(13)]
    one_shot = mb.delay_and_sum([samples], [2.0])

    history = None
    streamed: list[float] = []
    for start in (0, 4, 8, 12):
        block = samples[start : start + 4]
        streamed.extend(mb.delay_and_sum([block], [2.0], history=history))
        boundary = mb.block_boundary(
            block,
            history[0] if history is not None else None,
            delay_samples=[2.0],
        )
        history = [boundary["history"]]

    assert streamed == one_shot


def test_trajectory_smoothing_tracks_constant_motion():
    tracker = mb.SourceTrajectory(alpha=0.5, beta=0.125, lookahead=1.0)
    angle = 0.0
    result = {}
    for _ in range(30):
        angle += 0.05
        result = tracker.update(angle)

    assert result["smoothed_angle"] == approx(angle, abs=0.06)
    assert result["pointing_angle"] == approx(angle, abs=0.08)


def test_polarity_detection_and_correction():
    source = [math.sin(i * 0.17) for i in range(256)]
    channels = [
        source,
        [-value for value in source],
        [0.92 * value for value in source],
    ]

    report = mb.correct_polarity(channels)

    assert report["inverted_indices"] == [1]
    assert mb.normalized_correlation(
        report["corrected"][0], report["corrected"][1]
    ) > 0.999
    assert mb.phase_invert(0.25) == -0.25


def test_temperature_sensitivity_compensation():
    report = mb.temperature_calibration(
        temperature=35.0,
        sensitivity_db=[0.0, 1.0],
        coefficient_db_per_c=[0.02, -0.03],
        n_mics=2,
    )

    assert report["drift_db"] == approx([0.2, 0.7])
    for drift_db, gain in zip(report["drift_db"], report["gains"]):
        assert drift_db + 20.0 * math.log10(gain) == approx(
            0.0, abs=1e-10
        )

    cal = mb.array_calibration(
        [0.0, 0.04],
        temperature=35.0,
        sensitivity_db=1.0,
        temp_coefficient_db_per_c=0.02,
    )
    expected = 10.0 ** (-1.2 / 20.0)
    assert cal["sensitivity_gains"][0] == approx(expected)


def test_endfire_beamform_recombines_propagated_impulse():
    # 近侧阵元先收到脉冲；远侧阵元的脉冲晚约 6 个采样。
    channels = [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    output = mb.beamform(
        channels,
        positions=[0.0, 0.043],
        direction=math.pi / 2.0,
        sample_rate=48_000,
    )

    assert max(output) > 0.98
    assert output[6] > 0.98


def test_determinism_and_supporting_contracts_are_stable():
    assert mb.determinism(11)["values"] == mb.determinism(11)["values"]
    assert mb.windowing(1) == [1.0]
    assert mb.validation(1)["ok"] is True
    assert mb.audit(1)["event_count"] == 1
    assert mb.fallback(1)["mode"].startswith("uniform")
    assert mb.metrics(1.0)["rms"] == 1.0
    assert mb.noise_floor(1.0)["noise_floor_rms"] == 1.0
