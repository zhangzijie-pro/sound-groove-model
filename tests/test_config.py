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
    assert cfg.model.post_ffn_hidden_dim == 1024
    assert cfg.loss.pit_pos_weight == 2.25
    assert cfg.loss.lambda_smooth == 0.03
    assert cfg.train.warmup_epochs == 5
    assert cfg.output.monitor == "der"
    assert cfg.validate.slot_threshold == 0.45
