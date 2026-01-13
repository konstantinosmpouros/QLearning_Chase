import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Parameters Optimized for Faster Convergence ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
ETA = 0.2  # Increased for faster exploration
BATCH_SIZE = 256
EPISODES = 50000

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)

class NFSPAgent:
    def __init__(self):
        self.rl_net = PolicyNetwork().to(device)
        self.sl_net = PolicyNetwork().to(device)
        self.optimizer_rl = optim.Adam(self.rl_net.parameters(), lr=LR)
        self.optimizer_sl = optim.Adam(self.sl_net.parameters(), lr=LR)
        self.replay_buffer = deque(maxlen=30000)
        self.sl_buffer = deque(maxlen=30000)

    def choose_action(self, state, train=True):
        with torch.no_grad():
            # If training, mix RL and SL. If testing, always use SL (Average Strategy)
            is_rl = train and random.random() < ETA
            net = self.rl_net if is_rl else self.sl_net
            probs = net(state)
            action = torch.multinomial(probs, 1).item()
            return action, ("RL" if is_rl else "SL")

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE: return
        
        # 1. RL Update (Best Response)
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        for s, a, r in batch:
            probs = self.rl_net(s)
            loss = -torch.log(probs[a] + 1e-8) * r
            self.optimizer_rl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rl_net.parameters(), 1.0)
            self.optimizer_rl.step()

        # 2. SL Update (Average Strategy / Fictitious Play)
        if len(self.sl_buffer) > BATCH_SIZE:
            batch = random.sample(self.sl_buffer, BATCH_SIZE)
            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch]).to(device)
            pred = self.sl_net(states)
            loss = nn.CrossEntropyLoss()(pred, actions)
            self.optimizer_sl.zero_grad()
            loss.backward()
            self.optimizer_sl.step()

# --- Environment Logic (Same as before) ---
class KuhnPoker:
    def __init__(self):
        self.cards = [0, 1, 2]
    def reset(self):
        self.deck = random.sample(self.cards, 3)
        self.p1_card, self.p2_card = self.deck[0], self.deck[1]
        self.history = ""
        return self.get_state(0)
    def get_state(self, player_idx):
        card = self.p1_card if player_idx == 0 else self.p2_card
        state = np.zeros(6); state[card] = 1
        if self.history == "P": state[3] = 1
        elif self.history == "B": state[4] = 1
        elif self.history == "PB": state[5] = 1
        return torch.FloatTensor(state).to(device)
    def step(self, action):
        self.history += "P" if action == 0 else "B"
        terminal, reward = self.check_terminal()
        if terminal: return None, reward
        return self.get_state(len(self.history) % 2), 0
    def check_terminal(self):
        h, p1, p2 = self.history, self.p1_card, self.p2_card
        if h == "PP": return True, 1 if p1 > p2 else -1
        if h == "PBB": return True, -1 
        if h == "PBP": return True, 1 if p1 > p2 else -1 
        if h == "BP": return True, 1 
        if h == "BB": return True, 2 if p1 > p2 else -2 
        return False, 0

# --- Training Execution ---
env = KuhnPoker()
agent = NFSPAgent()

print(f"Training for {EPISODES} episodes on {device}...")

for ep in range(EPISODES):
    state = env.reset()
    game_history = []
    curr_state = state
    while curr_state is not None:
        action, source = agent.choose_action(curr_state)
        if source == "RL": agent.sl_buffer.append((curr_state, action))
        game_history.append((curr_state, action))
        curr_state, reward = env.step(action)
    for s, a in game_history:
        agent.replay_buffer.append((s, a, reward))
    agent.train()

    if ep % 5000 == 0:
        print(f"\n--- Episode {ep} Strategy Report ---")
        for val, name in {0: "Jack", 1: "Queen", 2: "King"}.items():
            test_s = torch.zeros(6).to(device); test_s[val] = 1
            with torch.no_grad():
                probs = agent.sl_net(test_s)
                print(f"{name:5}: Check {probs[0]:.2f} | Bet {probs[1]:.2f}")

print("\n--- Final Equilibrium Check ---")
for val, name in {0: "Jack", 1: "Queen", 2: "King"}.items():
    test_s = torch.zeros(6).to(device); test_s[val] = 1
    with torch.no_grad():
        probs = agent.sl_net(test_s)
        print(f"{name:5}: Check {probs[0]:.2f} | Bet {probs[1]:.2f}")