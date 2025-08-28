import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.monitor import Monitor

# 0) Registrar ALE (para habilitar el namespace "ALE")
gym.register_envs(ale_py)

# 1) Crear el entorno SIN frameskip (lo hace el wrapper)
env = gym.make("ALE/Breakout-v5", render_mode=None, frameskip=1)

# 2) Preprocesado Atari + stack de 4 frames (reduce memoria)
env = AtariPreprocessing(
    env,
    screen_size=84,
    grayscale_obs=True,
    frame_skip=4,                # frameskip del wrapper
    noop_max=30,
    terminal_on_life_loss=False,
    scale_obs=False,
)
env = FrameStackObservation(env, stack_size=4)  
env = Monitor(env, filename="monitor.csv")
# 3) DQN con buffer
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=100_000,
    learning_starts=10_000,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.10,
    exploration_final_eps=0.01,
)

# 4) Entrenar
model.learn(total_timesteps=20_000_000)

# 5) Cerrar y guardar
env.close()
model.save("dqn_breakout")



