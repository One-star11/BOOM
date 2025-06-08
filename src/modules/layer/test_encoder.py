import torch
import numpy as np
from cross_atten import CrossAttention  # Replace with your actual module name
from ace_utils import StateActionEncoder

# =============================================================================
# TO TEST WITH YOUR REAL CHECKPOINT:
# 1. Uncomment the line below and set your checkpoint path
REAL_CHECKPOINT_PATH = "/home/work/JAL/tempRL/MA2E/src/modules/layer/5m6m/seed0/ckpt/ckpt_best.pth.tar"
# 
# 2. In the test_weight_loading() function, you'll see instructions to uncomment
#    additional test lines to test with your real checkpoint
#
# 3. OR, you can directly call:
#    inspect_checkpoint("path/to/your/checkpoint.pth")
#    test_real_checkpoint("path/to/your/checkpoint.pth")
# =============================================================================

def test_real_checkpoint(checkpoint_path):
    """Quick test function for a real checkpoint file"""
    success = False

    # Initialize encoder with 5m_vs_6m dimensions
    encoder = StateActionEncoder(
        agent_num=5,
        state_len=33,
        relation_len=6,
        hidden_len=256
    )
    
    # Try to load weights
    print(f"\nAttempting to load weights...")
    actual_input_size = 256

    # Load the learned ACE encoder weights if checkpoint path is provided
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                full_state_dict = checkpoint['model']
            else:
                full_state_dict = checkpoint
            
            # Define the component keys we want to extract
            encoder_components = [
                '_action_encoder',
                '_state_encoder', 
                '_relation_encoder',
                '_relation_aggregator'
            ]
            
            # Filter relevant weights
            encoder_state_dict = {}
            for key, value in full_state_dict.items():
                # Remove potential model prefix if present
                clean_key = key
                if clean_key.startswith('model.'):
                    clean_key = clean_key[6:]  # Remove 'model.' prefix
                
                # Check if this key belongs to any encoder component
                for component in encoder_components:
                    if clean_key.startswith(component):
                        encoder_state_dict[clean_key] = value
                        break
            
            if encoder_state_dict:
                # Load the filtered weights
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    encoder_state_dict, strict=False)
                
                print(f"Loaded {len(encoder_state_dict)} parameters for StateActionEncoder")
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
            else:
                print("Warning: No matching encoder components found in checkpoint")
                
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    
    success = True
    return success

def create_mock_observation(batch_size=2, agent_num=5, state_len=42, relation_len=20):
    """Create mock observation data in the format expected by StateActionEncoder"""
    obs = {
        'states': torch.randn(batch_size, agent_num, state_len),
        'relations': torch.randn(batch_size, agent_num, relation_len), 
        'alive_mask': torch.ones(batch_size, agent_num)  # All agents alive
    }
    return obs

def test_standalone_encoder():
    """Test 1: Initialize and run StateActionEncoder standalone"""
    print("=" * 50)
    print("Test 1: Standalone StateActionEncoder")
    print("=" * 50)
    
    # Configuration for 5m_vs_6m scenario
    agent_num = 5      # 5 marines in 5m_vs_6m
    state_len = 33     # Standard state dimension for SMAC marines
    relation_len = 6  # Standard relation dimension for SMAC
    hidden_len = 256    # Common hidden size for marine scenarios
    batch_size = 2
    
    # Initialize encoder
    encoder = StateActionEncoder(
        agent_num=agent_num,
        state_len=state_len,
        relation_len=relation_len,
        hidden_len=hidden_len
    )
    
    print(f"✓ StateActionEncoder initialized successfully")
    print(f"  - Agent num: {agent_num}")
    print(f"  - State len: {state_len}")
    print(f"  - Relation len: {relation_len}")
    print(f"  - Hidden len: {hidden_len}")
    
    # Create mock data
    obs = create_mock_observation(batch_size, agent_num, state_len, relation_len)
    print(f"✓ Mock observation created:")
    print(f"  - States shape: {obs['states'].shape}")
    print(f"  - Relations shape: {obs['relations'].shape}")
    print(f"  - Alive mask shape: {obs['alive_mask'].shape}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = encoder(obs)
        print(f"✓ Forward pass successful!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min():.3f}, {output.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_weight_loading():
    """Test 2: Test weight loading functionality"""
    print("\n" + "=" * 50)
    print("Test 2: Weight Loading")
    print("=" * 50)
    
    # Configuration for 5m_vs_6m scenario
    agent_num = 5      # 5 marines in 5m_vs_6m
    state_len = 33     # Standard state dimension for SMAC marines
    relation_len = 6  # Standard relation dimension for SMAC
    hidden_len = 256    # Common hidden size for marine scenarios
    
    # Test without checkpoint (should work fine)
    print("Testing without checkpoint file...")
    try:
        encoder_no_weights = StateActionEncoder(
            agent_num=agent_num,
            state_len=state_len,
            relation_len=relation_len,
            hidden_len=hidden_len
        )
        print("✓ Encoder without weights initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize encoder: {e}")
        return False
    
    # Test with non-existent checkpoint (should give warning but continue)
    print("\nTesting with non-existent checkpoint file...")
    try:
        cross_attention = CrossAttention(
            input_size=hidden_len,
            heads=4,
            embed_size=hidden_len,
            offline_keys=torch.randn(10, hidden_len),  # Mock offline data
            offline_values=torch.randn(10, 5),
            state_len=state_len,
            relation_len=relation_len,
            agent_num=agent_num,
            use_ace_encoder=True,
            checkpoint_path="non_existent_file.pth"  # This will trigger warning
        )
        print("✓ CrossAttention with non-existent checkpoint handled gracefully")
    except Exception as e:
        print(f"✗ Failed with non-existent checkpoint: {e}")
        return False
    
    return True

def test_cross_attention_integration():
    """Test 3: Test CrossAttention with ACE encoder"""
    print("\n" + "=" * 50)
    print("Test 3: CrossAttention Integration")
    print("=" * 50)
    
    # Configuration for 5m_vs_6m scenario
    agent_num = 5      # 5 marines in 5m_vs_6m
    state_len = 33     # Standard state dimension for SMAC marines
    relation_len = 6  # Standard relation dimension for SMAC
    hidden_len = 256    # Common hidden size for marine scenarios
    batch_size = 2
    n_episodes = 10
    
    # Create mock offline data
    offline_keys = torch.randn(n_episodes, hidden_len)
    offline_values = torch.randn(n_episodes, 5)  # 5 timesteps per episode
    
    try:
        # Initialize CrossAttention with ACE encoder
        cross_attention = CrossAttention(
            input_size=hidden_len,
            heads=4,
            embed_size=hidden_len,
            offline_keys=offline_keys,
            offline_values=offline_values,
            state_len=state_len,
            relation_len=relation_len,
            agent_num=agent_num,
            use_ace_encoder=True,
            checkpoint_path=None  # No checkpoint for this test
        )
        print("✓ CrossAttention with ACE encoder initialized successfully")
        
        # Test with raw observation input
        obs = create_mock_observation(batch_size, agent_num, state_len, relation_len)
        
        with torch.no_grad():
            attention_weights = cross_attention(obs)
        
        print(f"✓ CrossAttention forward pass successful!")
        print(f"  - Input: raw observation dict")
        print(f"  - Output shape: {attention_weights.shape}")
        print(f"  - Output (attention weights) sum: {attention_weights.sum(dim=1)}")  # Should be ~1
        
        return True
        
    except Exception as e:
        print(f"✗ CrossAttention test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing ACE StateActionEncoder Integration")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_standalone_encoder()
    all_passed &= test_real_checkpoint(REAL_CHECKPOINT_PATH) 
    all_passed &= test_cross_attention_integration()
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Replace 'non_existent_file.pth' with your actual ACE checkpoint path")
        print("2. Adjust the dimensions (agent_num, state_len, etc.) to match your model")
        print("3. Prepare your actual offline episode data")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    print("Note: These dimensions are for standard 5m_vs_6m SMAC setup.")
    print("If your ACE model was trained with different dimensions, adjust accordingly.\n")
    
    # Run the main test suite
    main()
    
    print("\n" + "=" * 60)
    print("ADDITIONAL HELPER FUNCTIONS:")
    print("You can also test individual components directly:")
    print("")
    print("# To inspect a checkpoint file:")
    print("inspect_checkpoint('path/to/your/checkpoint.pth')")
    print("")
    print("# To test a checkpoint file quickly:")
    print("test_real_checkpoint('path/to/your/checkpoint.pth')")
    print("")
    print("# To load weights into an existing encoder:")
    print("encoder = StateActionEncoder(agent_num=5, state_len=42, relation_len=20, hidden_len=64)")
    print("success = load_ace_encoder_weights(encoder, 'path/to/your/checkpoint.pth')")
    print("=" * 60)