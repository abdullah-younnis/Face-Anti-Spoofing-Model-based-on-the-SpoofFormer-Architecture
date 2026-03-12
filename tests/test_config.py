"""Tests for configuration dataclasses."""

import pytest


def test_model_config_defaults():
    """Test ModelConfig default values."""
    from spoofformer.config import ModelConfig
    
    config = ModelConfig()
    assert config.img_size == 224
    assert config.patch_size == 16
    assert config.in_channels == 3
    assert config.embed_dim == 768
    assert config.num_heads == 12
    assert config.num_layers == 12
    assert config.mlp_ratio == 4.0
    assert config.dropout == 0.1
    assert config.attention_dropout == 0.0
    assert config.extract_layers == [8, 11]


def test_model_config_vit_small():
    """Test ViT-Small factory method."""
    from spoofformer.config import ModelConfig
    
    config = ModelConfig.vit_small()
    assert config.embed_dim == 384
    assert config.num_heads == 6
    assert config.num_layers == 12


def test_model_config_vit_base():
    """Test ViT-Base factory method."""
    from spoofformer.config import ModelConfig
    
    config = ModelConfig.vit_base()
    assert config.embed_dim == 768
    assert config.num_heads == 12
    assert config.num_layers == 12


def test_model_config_patch_size_validation():
    """Test that img_size must be divisible by patch_size."""
    from spoofformer.config import ModelConfig
    
    with pytest.raises(ValueError, match="img_size.*must be divisible by patch_size"):
        ModelConfig(img_size=225, patch_size=16)


def test_model_config_head_dimension_validation():
    """Test that embed_dim must be divisible by num_heads."""
    from spoofformer.config import ModelConfig
    
    with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
        ModelConfig(embed_dim=100, num_heads=12)


def test_model_config_extract_layers_validation():
    """Test that extract_layers values must be less than num_layers."""
    from spoofformer.config import ModelConfig
    
    with pytest.raises(ValueError, match="extract_layers.*must be less than num_layers"):
        ModelConfig(num_layers=12, extract_layers=[8, 12])


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    from spoofformer.config import TrainingConfig
    
    config = TrainingConfig()
    assert config.data_root == "dataset"
    assert config.batch_size == 32
    assert config.num_workers == 4
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.05
    assert config.epochs == 100
    assert config.warmup_epochs == 5
    assert config.bce_weight == 1.0
    assert config.intermediate_weight == 0.5
    assert config.checkpoint_dir == "checkpoints"
    assert config.save_every == 5
    assert config.seed == 42


def test_training_config_learning_rate_validation():
    """Test that learning_rate must be positive."""
    from spoofformer.config import TrainingConfig
    
    with pytest.raises(ValueError, match="learning_rate.*must be positive"):
        TrainingConfig(learning_rate=-0.001)
    
    with pytest.raises(ValueError, match="learning_rate.*must be positive"):
        TrainingConfig(learning_rate=0)


def test_training_config_batch_size_validation():
    """Test that batch_size must be a positive integer."""
    from spoofformer.config import TrainingConfig
    
    with pytest.raises(ValueError, match="batch_size.*must be a positive integer"):
        TrainingConfig(batch_size=0)
    
    with pytest.raises(ValueError, match="batch_size.*must be a positive integer"):
        TrainingConfig(batch_size=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
