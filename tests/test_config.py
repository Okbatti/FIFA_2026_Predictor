from predictor import config

def test_paths_exist():
    assert config.ROOT.exists()
    assert config.ARTIFACTS.exists()

def test_defaults_sane():
    assert 0.0 <= config.BLEND_WEIGHT_DEFAULT <= 1.0
    assert config.SIM_N >= 1000  # enough Monte Carlo draws for ~1% title-odds resolution
    assert config.MAX_GOALS_GRID >= 8
