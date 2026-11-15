import pytest
import torch
from behavior_helix import BehaviorHELIX, create_behavior_helix

class TestBehaviorHELIX:
    def test_init(self):
        m = BehaviorHELIX(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = BehaviorHELIX(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_behavior_helix()
        assert isinstance(m, BehaviorHELIX)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
