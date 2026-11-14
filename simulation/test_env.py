import numpy as np
from isac_env import CellFreeISAC3DEnv

def test_environment():
    print("Starting the brute-force stress test of the environment...")
    env = CellFreeISAC3DEnv(M=2, N=2, K=15)
    
    state = env.reset()
    print(f"Reset succeeded. Initial state dimension: {state.shape}")
    
    # Run 5 arbitrary steps to see whether it crashes
    for step in range(5):
        # Generate a fully random continuous action in [-1, 1]
        random_action = np.random.uniform(-1, 1, env.action_space.shape[0])
        
        next_state, reward, done, info = env.step(random_action)
        
        print(f"Step {step+1}:")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Communication rate sum_rate: {info['sum_rate']:.4f}")
        print(f"  - Sensing accuracy fim_det: {info['fim_det']:.4f}")
        print(f"  - The 4 selected APs: {info['selected_aps']}")
        
        if np.isnan(reward) or np.isinf(reward):
            print("Warning: the reward is NaN or infinite!")
            break

if __name__ == '__main__':
    test_environment()