import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DroneDeliveryEnv(gym.Env):
    """
    Improved Drone Delivery Optimization MDP Environment.
    More liberal rewards and penalties.
    """

    def __init__(
        self,
        m, n,
        customer_locations,
        drone_capacities,
        drone_max_ranges,
        w,
        alpha=0.7, beta=0.9, phi=2.2,
        verbose=True
    ):
        super().__init__()
        self.m, self.n, self.w  = m, n, w
        self.alpha, self.beta, self.phi = alpha, beta, phi
        self.verbose = verbose

        self.customer_locations = np.asarray(customer_locations, dtype=np.float32)
        self.drone_capacities   = np.asarray(drone_capacities , dtype=np.float32)
        self.drone_max_ranges   = np.asarray(drone_max_ranges , dtype=np.float32)

        self.d_depot = np.linalg.norm(self.customer_locations, axis=1)

        # --------- dynamic state ---------
        self.delta     = np.zeros(m, dtype=np.float32)   # delivered flags
        self.load      = np.zeros(n, dtype=np.float32)   # drone load
        self.eta       = np.zeros(n, dtype=np.int32)     # customers per drone
        self.pos       = np.zeros((n, 2), dtype=np.float32)   # last location for each drone
        self.dist_flown= np.zeros(n, dtype=np.float32)   # km flown per drone
        self.t         = 0                               # current customer index
        self.active    = np.zeros(n, dtype=np.int32)     # 1 if drone is used
        self.delivery_path = [[] for _ in range(n)]      # track delivery sequence for each drone

        # --------- observation space ---------
        low  = np.concatenate([np.zeros(m), np.zeros(n), np.zeros(n)])
        high = np.concatenate([np.ones(m), self.drone_capacities, np.full(n, m)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(n + 1)   # 0…n-1 assign, n = new-drone

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _range_at_load(self, idx, load):
        return self.drone_max_ranges[idx] * (1 - self.alpha * min(load / self.drone_capacities[idx], self.beta))

    def reset(self, *, seed=None, options=None):
        self.delta.fill(0)
        self.load.fill(0)
        self.eta.fill(0)
        self.pos.fill(0.)
        self.dist_flown.fill(0.)
        self.t = 0
        self.active.fill(0)
        self.delivery_path = [[] for _ in range(self.n)]
        self._log("\n=== RESET ENV ===")
        obs = np.concatenate([self.delta, self.load, self.eta.astype(np.float32)])
        info = {}
        return obs, info

    def step(self, action):
        done, reward = False, 0.0
        self._log(f"\n[STEP] cid={self.t}  action={action}")

        if self.t >= self.m:
            done   = True
            # (was 20 - 2*num_drones, now 30 - 1*num_drones)
            reward = 30.0 - 1.0 * np.sum(self.eta > 0)
            self._log(f"All customers served. Terminal reward {reward:.1f}")
            obs = np.concatenate([self.delta, self.load, self.eta.astype(np.float32)])
            info = {}
            return obs, reward, done, False, info

        cid   = self.t
        c_xy  = self.customer_locations[cid]
        d2dep = self.d_depot[cid]

        if action < self.n:
            if self.active[action] == 0:
                reward = -2  # -5 now -2
                self._log("✗ Tried assigning to non-existent drone → −1")
                self.t += 1
            else:
                reward = self._attempt_assign(drone_idx=action, cust_idx=cid, cust_xy=c_xy, d_home=d2dep)
        else:  # action == n  → create new drone for customer
            idle = np.where(self.active == 0)[0]
            if idle.size == 0:
                reward = -5  #  -10 now -5
                self._log("✗ Tried to create new drone but max reached → −2")
                self.t += 1
            else:
                reward = self._attempt_assign(drone_idx=idle[0], cust_idx=cid, cust_xy=c_xy, d_home=d2dep, fresh=True)

        # ------------- terminal check -------------
        if self.t >= self.m and not done:
            done   = True
            reward = 30.0 - 1.0 * np.sum(self.eta > 0)
            self._log(f"All customers served. Terminal reward {reward:.1f}")

        obs = np.concatenate([self.delta, self.load, self.eta.astype(np.float32)])
        info = {}
        return obs, reward, done, False, info

    def _attempt_assign(self, *, drone_idx, cust_idx, cust_xy, d_home, fresh=False):
        k = drone_idx
        seg = np.linalg.norm(cust_xy - self.pos[k])  # distance from current drone pos to customer
        new_load = self.load[k] + self.w
        to_customer = seg
        to_depot    = np.linalg.norm(cust_xy - np.array([0, 0]))
        total_trip  = to_customer + to_depot
        range_left = self._range_at_load(k, new_load) - self.dist_flown[k]
        need = self.phi * d_home

        if new_load > self.drone_capacities[k]:
            self._log(f"✗ Infeasible: over capacity ({new_load:.2f} > {self.drone_capacities[k]:.2f}) → −1")
            self.t += 1
            return -1   # -3, now -1

        if range_left < need:
            self._log(f"✗ Infeasible: not enough range (need {need:.2f}, left {range_left:.2f}) → −1")
            self.t += 1
            return -1   # -3, now -1

        # --- feasible assignment ---
        self.load[k]         = new_load
        self.eta[k]         += 1
        self.delta[cust_idx] = 1
        self.pos[k]          = cust_xy
        self.dist_flown[k]  += to_customer
        self.active[k]       = 1
        self.delivery_path[k].append(int(cust_idx))
        self.t              += 1

        if fresh:
            self._log(f"Created new drone {k} for delivery. Reward +2")
            return +2   #+1, now +2
        else:
            self._log(f"Assigned to existing drone {k}. Reward +0.5")
            return -0.5 # -1 now 0.5

    def render(self, mode='human'):
        print(f"[Render] t={self.t}/{self.m}  load={self.load}  pos={self.pos}  eta={self.eta}  active={self.active}")
        for i, path in enumerate(self.delivery_path):
            if path:
                print(f"Drone {i}: path {path}, total_dist_flown={self.dist_flown[i]:.2f}")