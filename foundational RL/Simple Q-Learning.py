import numpy as np

# Environment (simple 1D world)
states = 5
actions = ["left", "right"]

# Q-table initialization
Q = np.zeros((states, len(actions)))

# Parameters
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration

def get_reward(state):
    return 1 if state == 4 else 0

# Training
for episode in range(1000):
    state = 0
    
    while state != 4:
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice([0,1])
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state = state + 1 if action == 1 else max(0, state - 1)
        
        reward = get_reward(next_state)
        
        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state

print("Learned Q-table:")
print(Q)

