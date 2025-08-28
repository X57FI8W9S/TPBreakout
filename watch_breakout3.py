import argparse
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import matplotlib.pyplot as plt

def make_env(render_mode: str):
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,              # ya que el entorno tiene frameskip=1
        noop_max=30,
        terminal_on_life_loss=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)   # stackea 4 frames
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dqn_breakout", help="Ruta sin .zip (SB3 lo agrega)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", choices=["human", "rgb_array"], default="human",
                        help="human = ventana nativa; rgb_array = mostrar último frame con matplotlib")
    parser.add_argument("--deterministic", action="store_true", help="Acciones determinísticas")
    args = parser.parse_args()

    # Registrar ALE
    gym.register_envs(ale_py)

    # Cargar entorno y modelo
    env = make_env(args.render)
    model = DQN.load(args.model, device="auto")

    # Episodios
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        final_frame = None  # para rgb_array

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

            # Renderizar
            frame = env.render()
            if args.render == "rgb_array":
                final_frame = frame

        print(f"Episodio {ep}: recompensa={ep_return}")

        # Mostrar último frame
        if args.render == "rgb_array" and final_frame is not None:
            plt.imshow(final_frame)
            plt.axis("off")
            plt.title(f"Breakout — último frame (episodio {ep})")
            plt.show()

    env.close()

if __name__ == "__main__":
    main()
