import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int)  # 0 empty, 1 agent, -1 opponent

    def reset(self):
        self.board[:] = 0
        return self.board.copy()

    def available_actions(self):
        return [(r,c) for r in range(3) for c in range(3) if self.board[r,c] == 0]

    def step(self, action, player):
        r,c = action
        self.board[r,c] = player
        reward, done = self.check_winner(player)
        return self.board.copy(), reward, done

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i,:] == player) or np.all(self.board[:,i] == player):
                return 1, True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return 1, True
        if not self.available_actions():
            return 0, True  # draw
        return 0, False

# --- Exploration Strategies ---
def epsilon_greedy(Q, state, actions, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(actions, key=lambda a: Q.get((tuple(state.flatten()), a), 0))

def softmax(Q, state, actions, tau=1.0):
    q_vals = np.array([Q.get((tuple(state.flatten()), a), 0) for a in actions])
    exp_q = np.exp(q_vals / tau)
    probs = exp_q / np.sum(exp_q)
    return actions[np.random.choice(len(actions), p=probs)]

# --- Simulation ---
def simulate(strategy, episodes=500):
    env = TicTacToe()
    Q = {}
    wins, steps = 0, 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            if strategy == "epsilon":
                action = epsilon_greedy(Q, state, actions, epsilon=0.1)
            else:
                action = softmax(Q, state, actions, tau=1.0)

            next_state, reward, done = env.step(action, player=1)
            steps += 1
            Q[(tuple(state.flatten()), action)] = reward
            state = next_state

            # Opponent random move
            if not done and env.available_actions():
                opp_action = random.choice(env.available_actions())
                _, _, done = env.step(opp_action, player=-1)

        if reward == 1:
            wins += 1

    return wins/episodes, steps

# --- Compare Strategies ---
eps_winrate, eps_steps = simulate("epsilon", episodes=1000)
soft_winrate, soft_steps = simulate("softmax", episodes=1000)

print("Performance Comparison:")
print(f"Epsilon-Greedy -> Win Rate: {eps_winrate:.2f}, Steps: {eps_steps}")
print(f"Softmax       -> Win Rate: {soft_winrate:.2f}, Steps: {soft_steps}")
