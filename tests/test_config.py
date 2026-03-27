from omegaconf import OmegaConf


def test_experiment_config_load():
    cfg = OmegaConf.load("configs/experiment.yaml")

    assert "run" in cfg
    assert "model" in cfg
    assert "loss" in cfg
    assert "data" in cfg
    assert "train" in cfg
    assert "validate" in cfg
    assert "output" in cfg
    assert cfg.model.embedding_dim == 256
    assert cfg.model.max_mix_speakers == 4
