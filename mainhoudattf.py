import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define the possible actions for the agent
ACTIONS = ['action A', 'action B']

# Define the Q-learning parameters
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.2 # Epsilon-greedy policy parameter

# Define the Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to choose an action using epsilon-greedy policy
def choose_action(state, q_net):
    if random.uniform(0, 1) < epsilon:
        # Randomly choose an action (exploration)
        return random.choice(ACTIONS)
    else:
        # Choose the best action based on Q-values (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = q_net(state_tensor)
            action_index = q_values.argmax().item()
            return ACTIONS[action_index]

# Function to simulate the network system and return the reward and new state
def simulate_network_system(state, action):
    # Implement the logic for the network system
    # Here, we just return a random reward and new state for demonstration purposes
    new_state = random.choice([10, 20])
    reward = random.randint(1, 10)
    return new_state, reward

# Q-learning algorithm
def q_learning(q_net, num_episodes):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(q_net.parameters(), lr=alpha)

    for episode in range(num_episodes):
        state = 0  # Initial state of the agent
        total_reward = 0

        while True:
            # Choose an action using the epsilon-greedy policy
            action = choose_action(state, q_net)

            # Simulate the network system and get the reward and new state
            new_state, reward = simulate_network_system(state, action)
            total_reward += reward

            # Convert states and actions to tensors
            state_tensor = torch.FloatTensor(state)
            new_state_tensor = torch.FloatTensor(new_state)

            # Calculate the target Q-value using the Q-network
            q_values = q_net(state_tensor)
            with torch.no_grad():
                new_q_values = q_net(new_state_tensor)
                max_q_value = new_q_values.max()

            target_q_value = reward + gamma * max_q_value

            # Update the Q-network using the Q-learning update rule
            optimizer.zero_grad()
            loss = criterion(q_values[ACTIONS.index(action)], target_q_value)
            loss.backward()
            optimizer.step()

            state = new_state

            if state == 20:
                # Reached the terminal state (end of episode)
                break

        print(f"Episode {episode+1}, Total reward: {total_reward}")

# Main function
if __name__ == "__main__":
    state_size = 2  # Change this according to your state representation
    action_size = len(ACTIONS)
    q_net = QNetwork(state_size, action_size)
    num_episodes = 1000
    q_learning(q_net, num_episodes)
