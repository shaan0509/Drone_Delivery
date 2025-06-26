import numpy as np, gymnasium as gym, time, pathlib, os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.envs.drone_delivery_env import DroneDeliveryEnv

DATA_FILE   = "data/drone_dataset.npz"
TIMESTEPS   = 100_000
TOTAL_STEPS = 1_000_000
RUN_ID      = str(int(time.time()))

LOGDIR   = pathlib.Path("logs")   / f"run_{RUN_ID}"
MODELDIR = pathlib.Path("models") / f"run_{RUN_ID}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)

with np.load(DATA_FILE, allow_pickle=True) as d:
    data = {k: d[k] for k in d.files}

N     = len(data["cust_xy"])
idxs  = np.arange(N)
np.random.default_rng(123).shuffle(idxs)
split = int(0.8 * len(idxs))
train_idxs, test_idxs = idxs[:split], idxs[split:]
print(f"Dataset: {len(idxs)} → train {len(train_idxs)} | test {len(test_idxs)}")

MAX_LEN = max(len(l) + 2*min(len(l), 10) for l in data["cust_xy"])
def _pad(vec, length=MAX_LEN):
    if vec.size == length:
        return vec
    out = np.zeros(length, dtype=np.float32); out[:vec.size] = vec
    return out

def make_env_from_row(i):
    locs = data["cust_xy"][i]
    p    = {k: data[k][i] for k in ["capacity","max_range","alpha","beta","phi"]}
    n_slots = min(len(locs), 10)
    return DroneDeliveryEnv(
        m=len(locs), n=n_slots,
        customer_locations=locs,
        drone_capacities=[p["capacity"]]*n_slots,
        drone_max_ranges=[p["max_range"]]*n_slots,
        w=5.0,
        alpha=p["alpha"], beta=p["beta"], phi=p["phi"],
        verbose=False
    )

MAX_LEN = max(len(l) + 2*min(len(l), 10) for l in data["cust_xy"])

def _pad(vec, length=MAX_LEN):
    out = np.zeros(length, dtype=np.float32)
    out[:min(vec.size, length)] = vec[:min(vec.size, length)]
    return out

class DatasetSamplerEnv(gym.Env):
    def __init__(self, idx_pool, seed=0, max_episode_steps=100):
        super().__init__()
        self.idx_pool = idx_pool
        self.rng = np.random.default_rng(seed)
        self.inner = None
        self._ep_rew = 0.0
        self._ep_len = 0
        self.max_episode_steps = max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(MAX_LEN,), dtype=np.float32
        )
        sample_env = make_env_from_row(self.idx_pool[0])
        self.action_space = sample_env.action_space

    def reset(self, **kw):
        if self._ep_len > 0:
            info = {"episode": {"r": self._ep_rew, "l": self._ep_len}}
        else:
            info = {}
        self._ep_rew, self._ep_len = 0.0, 0
        kw.pop("seed", None)
        self.inner = make_env_from_row(self.rng.choice(self.idx_pool))
        obs, _ = self.inner.reset(**kw)
        return _pad(obs), info

    def step(self, action):
        obs, r, done, truncated, info = self.inner.step(action)
        self._ep_rew += r
        self._ep_len += 1
        if self._ep_len >= self.max_episode_steps:
            done = True
        return _pad(obs), r, bool(done), False, info

train_env = Monitor(DatasetSamplerEnv(train_idxs, seed=321))

TEST_SAMPLE = 100
sampled_test = np.random.choice(test_idxs, TEST_SAMPLE, replace=False)

def make_test_factory(idx, seed=0):
    def _init():
        return Monitor(DatasetSamplerEnv([idx], seed=seed))
    return _init

test_factories = [make_test_factory(i) for i in sampled_test]
eval_vec_env   = DummyVecEnv(test_factories)

eval_callback = EvalCallback(
        eval_vec_env,
        eval_freq        = TIMESTEPS,
        deterministic    = True,
        n_eval_episodes  = TEST_SAMPLE,
        verbose          = 1
)

model = PPO(
    "MlpPolicy",
    train_env,
    device="cpu",
    tensorboard_log=str(LOGDIR),
    n_steps=4096,
    batch_size=2048,
    learning_rate=2.5e-4,
    gamma=0.99,
    verbose=1,
)

steps_done = 0
while steps_done < TOTAL_STEPS:
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name="PPO_drone",
                callback=eval_callback)
    steps_done += TIMESTEPS
    ckpt = MODELDIR / f"ppo_drone_{steps_done//1000}k"
    model.save(ckpt); print("Checkpoint →", ckpt)

final_path = MODELDIR / "ppo_drone_final"
model.save(final_path); print("Training complete. Model:", final_path)