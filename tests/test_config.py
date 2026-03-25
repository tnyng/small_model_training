from pathlib import Path

from modernbert_uq.config import load_config


def test_load_config_exposes_core_training_fields():
    config = load_config(Path("configs/default.yaml"))

    assert config.model_name
    assert config.max_length > 0
    assert config.train_batch_size > 0
    assert config.eval_batch_size > 0
    assert config.learning_rate > 0
    assert config.mc_dropout_passes > 0
    assert config.output_dir
