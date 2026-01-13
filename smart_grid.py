import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ENVIRONMENT (The Smart Grid Market)
# ==========================================
class SmartGridEnvironment:
    def __init__(self):
        # Simulation Parameters
        self.battery_capacity = 10
        self.market_impact_factor = 2.0  # How much 1 unit of action changes price
        
        # Base Price Curve (The "Duck Curve")
        # Low at night/mid-day, High in morning/evening
        self.base_prices = np.array([
            10, 10, 10, 10, 15, 20, 30, 40,  # 00:00 - 07:00 (Night/Morning)
            30, 20, 10, 10, 10, 10, 20, 30,  # 08:00 - 15:00 (Mid-day Solar dip)
            50, 80, 90, 80, 60, 40, 20, 10   # 16:00 - 23:00 (Evening Peak)
        ])

    def get_price(self, hour, net_action):
        """
        Price = Base_Price + (Impact * Net_Action)
        If everyone sells (negative action), price drops.
        If everyone buys (positive action), price spikes.
        """
        current_base = self.base_prices[hour]
        # Net action is sum of all agents' actions (e.g., +1 +1 = +2 demand)
        realized_price = current_base + (self.market_impact_factor * net_action)
        return max(0, realized_price)  # Price cannot be negative

    def step(self, current_charge, action):
        """
        Logic for Battery Physics.
        Action: 0=Idle, 1=Charge, 2=Discharge
        """
        # Map action index to change in charge
        # 0 -> 0 (Idle), 1 -> +1 (Charge), 2 -> -1 (Discharge)
        change = 0
        if action == 1: change = 1
        elif action == 2: change = -1
        
        # Check physical constraints
        next_charge = current_charge + change
        
        # If invalid move (overcharge or over-discharge), force Idle
        if next_charge < 0 or next_charge > self.battery_capacity:
            change = 0
            next_charge = current_charge
            
        return next_charge, change

# ==========================================
# 2. THE AGENT (RL + Fictitious Play Tracker)
# ==========================================
class FictitiousPlayAgent:
    def __init__(self, id, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.id = id
        self.alpha = alpha      # Learning Rate
        self.gamma = gamma      # Discount Factor
        self.epsilon = epsilon  # Exploration Rate
        
        # State: [Hour (0-23), BatteryLevel (0-10)]
        # Action: [0: Idle, 1: Charge, 2: Discharge]
        self.q_table = np.zeros((24, 11, 3))
        
        # FICTITIOUS PLAY COMPONENT:
        # We track the "Average Strategy" (The Nash Equilibrium candidate)
        # Counts: [Hour, BatteryLevel, Action]
        self.strategy_counts = np.zeros((24, 11, 3)) + 1e-5

    def get_action(self, hour, battery, use_average=False):
        """
        During Training: Uses Epsilon-Greedy on Q-Table (Best Response).
        During Evaluation: Uses the Average Strategy (Fictitious Play result).
        """
        if use_average:
            # Return the action with highest historical frequency
            return np.argmax(self.strategy_counts[hour, battery])
        
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
        
        # Exploitation (Best Response)
        return np.argmax(self.q_table[hour, battery])

    def update_q(self, s, a, r, s_next):
        """Standard Q-Learning Update"""
        h, b = s
        h_next, b_next = s_next
        
        best_next_a = np.argmax(self.q_table[h_next, b_next])
        td_target = r + self.gamma * self.q_table[h_next, b_next, best_next_a]
        td_error = td_target - self.q_table[h, b, a]
        
        self.q_table[h, b, a] += self.alpha * td_error

    def update_strategy(self, hour, battery, action):
        """Fictitious Play: Track what we actually did to form the average policy"""
        self.strategy_counts[hour, battery, action] += 1

# ==========================================
# 3. THE SIMULATION LOOP (Training)
# ==========================================

# Initialize
env = SmartGridEnvironment()
agent_A = FictitiousPlayAgent(id="A", gamma=0.99, alpha=0.1)
agent_B = FictitiousPlayAgent(id="B", gamma=0.99, alpha=0.1)

episodes = 50000  # Increased for better convergence
print("Starting Training (Long-Term Equilibrium)...")

for episode in range(episodes):
    bat_A, bat_B = 5, 5  # Reset batteries
    
    # 2. Linear Epsilon Decay (Start exploring, slowly move to exploitation)
    current_eps = max(0.01, 0.2 * (1 - episode / (episodes * 0.8)))
    agent_A.epsilon = current_eps
    agent_B.epsilon = current_eps

    for hour in range(24):
        state_A, state_B = (hour, bat_A), (hour, bat_B)
        
        # Get actions
        act_A = agent_A.get_action(hour, bat_A)
        act_B = agent_B.get_action(hour, bat_B)
        
        # Env step
        next_bat_A, change_A = env.step(bat_A, act_A)
        next_bat_B, change_B = env.step(bat_B, act_B)
        
        # Market price calculation
        price = env.get_price(hour, change_A + change_B)
        
        # Profit Calculation
        reward_A = -1 * change_A * price
        reward_B = -1 * change_B * price
        
        # Update Q-values and Fictitious Play history
        next_hour = (hour + 1) % 24
        agent_A.update_q(state_A, act_A, reward_A, (next_hour, next_bat_A))
        agent_B.update_q(state_B, act_B, reward_B, (next_hour, next_bat_B))
        agent_A.update_strategy(hour, bat_A, act_A)
        agent_B.update_strategy(hour, bat_B, act_B)
        
        bat_A, bat_B = next_bat_A, next_bat_B
    
    if episode % 1000 == 0:
        print(f"Progress: {int(episode/episodes*100)}% | Epsilon: {current_eps:.3f}")

print("Training Finished!")

# ==========================================
# 4. VISUALIZATION (The Equilibrium)
# ==========================================

# To visualize the Equilibrium, we run one "Test Day" 
# using the agents' LEARNED AVERAGE STRATEGY (not Q-values).

test_prices = []
test_bat_A = []
test_bat_B = []
actions_A = []
actions_B = []

bat_A = 5
bat_B = 5

for hour in range(24):
    # Use use_average=True to simulate the Nash Equilibrium behavior
    act_A = agent_A.get_action(hour, bat_A, use_average=True)
    act_B = agent_B.get_action(hour, bat_B, use_average=True)
    
    next_bat_A, change_A = env.step(bat_A, act_A)
    next_bat_B, change_B = env.step(bat_B, act_B)
    
    net_action = change_A + change_B
    price = env.get_price(hour, net_action)
    
    test_prices.append(price)
    test_bat_A.append(bat_A)
    test_bat_B.append(bat_B)
    actions_A.append(change_A)
    actions_B.append(change_B)
    
    bat_A = next_bat_A
    bat_B = next_bat_B

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Prices
ax1.plot(env.base_prices, 'k--', label="Base Price (No Agents)", alpha=0.5)
ax1.plot(test_prices, 'r-', linewidth=2, label="Equilibrium Price")
ax1.set_title("Market Price Stability (Equilibrium)")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True)

# Plot 2: Agent Strategies
hours = np.arange(24)
ax2.plot(hours, test_bat_A, 'b-o', label="Agent A Battery")
ax2.plot(hours, test_bat_B, 'g-s', label="Agent B Battery")
ax2.set_title("Agent Storage Strategies")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Battery Level")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()