"""Tests for model components."""

import pytest
import torch


def test_patch_embedding_output_shape():
    """Test PatchEmbedding output shape."""
    from spoofformer.models import PatchEmbedding
    
    embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)
    out = embed(x)
    
    # num_patches = (224/16)^2 = 196, +1 for CLS token = 197
    assert out.shape == (2, 197, 768)


def test_patch_embedding_validation():
    """Test PatchEmbedding validates img_size divisibility."""
    from spoofformer.models import PatchEmbedding
    
    with pytest.raises(ValueError, match="img_size.*must be divisible"):
        PatchEmbedding(img_size=225, patch_size=16)


def test_transformer_block():
    """Test TransformerBlock preserves shape."""
    from spoofformer.models import TransformerBlock
    
    block = TransformerBlock(embed_dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)
    out = block(x)
    
    assert out.shape == x.shape


def test_transformer_encoder():
    """Test TransformerEncoder output shape."""
    from spoofformer.models import TransformerEncoder
    
    encoder = TransformerEncoder(
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        extract_layers=[8, 11]
    )
    x = torch.randn(2, 197, 768)
    out = encoder(x, return_intermediate=False)
    
    assert out.shape == x.shape


def test_transformer_encoder_intermediate():
    """Test TransformerEncoder intermediate feature extraction."""
    from spoofformer.models import TransformerEncoder
    
    encoder = TransformerEncoder(
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        extract_layers=[8, 11]
    )
    x = torch.randn(2, 197, 768)
    out, intermediate = encoder(x, return_intermediate=True)
    
    assert out.shape == x.shape
    assert 8 in intermediate
    assert 11 in intermediate
    assert intermediate[8].shape == (2, 768)
    assert intermediate[11].shape == (2, 768)


def test_classification_head():
    """Test ClassificationHead output shape."""
    from spoofformer.models import ClassificationHead
    
    head = ClassificationHead(embed_dim=768, hidden_dim=256)
    x = torch.randn(2, 197, 768)
    out = head(x)
    
    assert out.shape == (2, 1)


def test_spoofformer_forward():
    """Test SpoofFormer forward pass."""
    from spoofformer.models import SpoofFormer
    
    model = SpoofFormer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        extract_layers=[4, 5]
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_intermediate=False)
    
    assert out.shape == (2, 1)


def test_spoofformer_intermediate():
    """Test SpoofFormer with intermediate features."""
    from spoofformer.models import SpoofFormer
    
    model = SpoofFormer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        extract_layers=[4, 5]
    )
    x = torch.randn(2, 3, 224, 224)
    out, intermediate = model(x, return_intermediate=True)
    
    assert out.shape == (2, 1)
    assert 4 in intermediate
    assert 5 in intermediate


def test_spoofformer_from_config():
    """Test SpoofFormer creation from config."""
    from spoofformer.models import SpoofFormer
    from spoofformer.config import ModelConfig
    
    config = ModelConfig.vit_small()
    config.extract_layers = [8, 11]
    model = SpoofFormer.from_config(config)
    
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    
    assert out.shape == (1, 1)


def test_liveness_score_range():
    """Test that sigmoid of logits is in [0, 1]."""
    from spoofformer.models import SpoofFormer
    
    model = SpoofFormer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        extract_layers=[4, 5]
    )
    model.eval()
    
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        scores = torch.sigmoid(logits)
    
    assert (scores >= 0).all()
    assert (scores <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
