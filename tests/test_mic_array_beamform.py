import pytest
import media.mic_array_beamform as mod

CASES = [
    ("array_calibration", (1,)),
    ("delay_weight", (1,)),
    ("block_boundary", (1,)),
    ("moving_source", (1,)),
    ("phase_invert", (1,)),
    ("determinism", (1,)),
    ("beamform", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("sample_rate", (1,)),
    ("metrics", (1,)),
    ("steer", (1,)),
    ("windowing", (1,)),
    ("noise_floor", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
