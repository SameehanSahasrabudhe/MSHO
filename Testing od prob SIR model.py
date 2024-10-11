# Testing code:

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Set random seed for reproducibility
np.random.seed(42)

# Model Parameters
CITY_RADIUS = 2.0          # Radius of the circular city
INFECTION_RADIUS = 0.1     # Infection radius
SIGMA = 0.03               # Standard deviation for Gaussian displacement
ALPHA = 0.5                # Probability of S -> I upon contact
BETA = 0.007               # Probability of I -> R
GAMMA = 0.0                # Probability of R -> S (0 means immunity is permanent)

# Simulation Parameters
NUM_AGENTS = 200           # Total number of agents
TIME_STEPS = 2000          # Total number of days to simulate
S0, I0, R0 = 198, 2, 0     # Initial counts of S, I, R

# State Encoding
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

def initialize_agents(num_agents, city_radius, S0, I0, R0):
    """
    Initialize agents uniformly within the circular city and assign initial states.
    """
    # Generate uniform distribution within a circle
    r = city_radius * np.sqrt(np.random.uniform(0, 1, num_agents))
    theta = np.random.uniform(0, 2 * np.pi, num_agents)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.vstack((x, y)).T

    # Assign states
    states = np.array([SUSCEPTIBLE]*S0 + [INFECTED]*I0 + [RECOVERED]*R0)
    np.random.shuffle(states)  # Shuffle to randomize initial infected
    return positions, states

def move_agents(positions, sigma, city_radius):
    """
    Move agents using a random walk and teleport if outside the city.
    """
    # Gaussian displacement
    displacements = np.random.normal(0, sigma, positions.shape)
    new_positions = positions + displacements

    # Compute distances from center
    distances = np.linalg.norm(new_positions, axis=1)

    # Find agents outside the city
    outside = distances > city_radius
    num_outside = np.sum(outside)
    if num_outside > 0:
        # Teleport outside agents to random positions inside the city
        r = city_radius * np.sqrt(np.random.uniform(0, 1, num_outside))
        theta = np.random.uniform(0, 2 * np.pi, num_outside)
        new_x = r * np.cos(theta)
        new_y = r * np.sin(theta)
        new_positions[outside] = np.vstack((new_x, new_y)).T

    return new_positions

def update_states(positions, states, alpha, beta, gamma, infection_radius):
    """
    Update the states of agents based on infection, recovery, and immunity loss.
    """
    # Separate indices by state
    susceptible_indices = np.where(states == SUSCEPTIBLE)[0]
    infected_indices = np.where(states == INFECTED)[0]
    recovered_indices = np.where(states == RECOVERED)[0]

    # Build KDTree for efficient neighbor search
    if len(infected_indices) > 0 and len(susceptible_indices) > 0:
        tree = KDTree(positions[infected_indices])
        # For each susceptible agent, find if any infected agent is within infection_radius
        neighbors = tree.query_ball_point(positions[susceptible_indices], infection_radius)
        # Determine which susceptible agents have at least one neighbor
        potentially_infected = np.array([len(n) > 0 for n in neighbors])
        # Susceptible agents that are close to at least one infected agent
        susceptible_close = susceptible_indices[potentially_infected]
        # Determine which of these become infected based on probability alpha
        new_infections = susceptible_close[
            np.random.uniform(0, 1, len(susceptible_close)) < alpha
        ]
        # Update states
        states[new_infections] = INFECTED

    # Infected agents recover with probability beta
    if len(infected_indices) > 0:
        recoveries = infected_indices[
            np.random.uniform(0, 1, len(infected_indices)) < beta
        ]
        states[recoveries] = RECOVERED

    # Recovered agents lose immunity with probability gamma
    if gamma > 0 and len(recovered_indices) > 0:
        lose_immunity = recovered_indices[
            np.random.uniform(0, 1, len(recovered_indices)) < gamma
        ]
        states[lose_immunity] = SUSCEPTIBLE

    return states

def simulate():
    # Initialize agents
    positions, states = initialize_agents(NUM_AGENTS, CITY_RADIUS, S0, I0, R0)

    # Lists to record daily counts
    S_counts = []
    I_counts = []
    R_counts = []

    for t in range(TIME_STEPS):
        # Move agents
        positions = move_agents(positions, SIGMA, CITY_RADIUS)

        # Update states based on interactions and transitions
        states = update_states(positions, states, ALPHA, BETA, GAMMA, INFECTION_RADIUS)

        # Record counts
        S_counts.append(np.sum(states == SUSCEPTIBLE))
        I_counts.append(np.sum(states == INFECTED))
        R_counts.append(np.sum(states == RECOVERED))

        # Optional: Print progress every 500 steps
        if (t+1) % 500 == 0:
            print(f"Day {t+1}: S={S_counts[-1]}, I={I_counts[-1]}, R={R_counts[-1]}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(S_counts, label='Susceptible', color='blue')
    plt.plot(I_counts, label='Infected', color='red')
    plt.plot(R_counts, label='Recovered', color='green')
    plt.xlabel('Day')
    plt.ylabel('Number of Agents')
    plt.title('SIR Model Simulation with Random Movement in a Circular City')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate()
