import numpy as np
import random

class CleaningRobotMDP:
    def __init__(self, grid_size=5, dirt_cells=None, obstacle_cells=None):
        self.grid_size = grid_size
        self.dirt_cells = dirt_cells if dirt_cells else [(1,1), (2,3), (4,4)]
        self.obstacle_cells = obstacle_cells if obstacle_cells else [(0,3), (3,1)]
        self.start_state = (0,0)
        self.state = self.start_state
        self.cleaned = set()

    def reset(self):
        self.state = self.start_state
        self.cleaned = set()
        return self.state

    def step(self, action):
        r, c = self.state
        if action == "up": r -= 1
        elif action == "down": r += 1
        elif action == "left": c -= 1
        elif action == "right": c += 1

        # Check boundaries
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return self.state, -1, False  # invalid move penalty

        new_state = (r, c)

        # Rewards
        if new_state in self.obstacle_cells:
            reward = -1
            new_state = self.state  # stay in place
        elif new_state in self.dirt_cells and new_state not in self.cleaned:
            reward = +1
            self.cleaned.add(new_state)
        else:
            reward = 0

        self.state = new_state
        done = len(self.cleaned) == len(self.dirt_cells)
        return new_state, reward, done

    def available_actions(self):
        return ["up", "down", "left", "right"]

# --- Policies ---
def random_policy(env, episodes=1):
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = random.choice(env.available_actions())
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            if done or steps > 50:  # stop after 50 steps
                print(f"Episode {ep+1}: Total Reward={total_reward}, Steps={steps}")
                break

# Example usage
env = CleaningRobotMDP()
random_policy(env, episodes=3)
