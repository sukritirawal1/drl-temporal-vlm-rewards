# drl-temporal-vlm-rewards
Intermediate Reward Computation using VLMs for Reinforcement Learning

To run PPO baseline:
python ppo.py --wandb_key [YOUR WANDB API KEY]

To change the environment type and the reward type, hardcode it in make_env
# Reference: https://minigrid.farama.org/content/training/
def make_env(log_dir):
    """Create and wrap the MiniGrid environment"""
    def _init():
        env = gym.make("MiniGrid-GoToObject-8x8-N2-v0", render_mode="rgb_array") # CHANGE THIS to change environment
        env = ImgObsWrapper(env)  # Convert dict obs to image obs
        env = Monitor(env, log_dir)  # Monitor for logging
        # env = CLIPRewardWrapper(env) # UNCOMMENT THIS TO ENABLE CUSTOM REWARDS
        return env
    return _init
