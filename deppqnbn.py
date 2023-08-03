import numpy as np

# Define the network system environment
class NetworkSystem:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.network = np.zeros((num_nodes, num_nodes)) + np.inf

    def add_edge(self, node1, node2, distance):
        self.network[node1][node2] = distance
        self.network[node2][node1] = distance

    def get_possible_actions(self, state):
        return [node for node in range(self.num_nodes) if node != state]

    def get_reward(self, state, action):
        return -self.network[state][action]

    def is_terminal_state(self, state, goal_state):
        return state == goal_state

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
            self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action])

# Main Q-learning loop
def q_learning(network_system, agent, start_state, goal_state, num_episodes=1000):
    for episode in range(num_episodes):
        state = start_state
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            reward = network_system.get_reward(state, action)
            next_state = action
            done = network_system.is_terminal_state(next_state, goal_state)
            agent.update_q_table(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

 # Create a network system with three nodes (0, 1, 2)
network_system = NetworkSystem(num_nodes=3)

    # Add edges with distances between nodes
network_system.add_edge(0, 1, 5)
network_system.add_edge(0, 2, 10)
network_system.add_edge(1, 2, 4)

    # Create the Q-learning agent
num_states = 3
num_actions = 2  # In this example, each node can choose to go to one of the other nodes
agent = QLearningAgent(num_states, num_actions)

    # Run Q-learning to find the shortest path from node 0 to node 2
start_state = 0
goal_state = 2
q_learning(network_system, agent, start_state, goal_state)

