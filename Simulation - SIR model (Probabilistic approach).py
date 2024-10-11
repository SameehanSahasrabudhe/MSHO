# Simulation by probabilities.

import numpy as np
import matplotlib.pyplot as plt

# By Probabilistic approach with boundaries. --------------------------------------------

#Agents
S0 = 198
I0 = 2 
R0 = 0   
agents_num = S0+I0

# Time parameters
tf = 1000
t0 = 0
dt = 1
num_steps = int((tf - t0)/dt)

#Model parameters
alpha, beta, gamma = 0.5, 0.007, 0

#City parameters
city_radius = 2
city = np.pi*(city_radius)**2
infection_radius = 0.1
sigma = 0.03    #Random walk parameter

def initialize_agents(agents_num, city_radius, S0, I0, R0):
    agents = []
    for _ in range(agents_num):
        r = np.random.uniform(0, 2*np.pi*city_radius)
        theta = np.random.uniform(0, 2*np.pi)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        agents.append([x, y])
    
    # Assign states
    S_agents = agents[:S0]
    I_agents = agents[S0:S0+I0]
    R_agents = agents[S0+I0:S0+I0+R0]

    # Shuffle initial infected agents for randomness
    np.random.shuffle(I_agents)
    return list(S_agents), list(I_agents), list(R_agents)
    

def movements(agents, sigma, city_radius):
    agents1 = np.array(agents)
    displacements = np.random.normal(0, sigma, agents1.shape)
    agents += displacements

    # If agents are going outside of the city then teleport to random location inside the city.
    for i, agent in enumerate(agents):
        if np.linalg.norm(agent) > city_radius:
            r = np.random.uniform(0, 2*np.pi*city_radius)
            theta = np.random.uniform(0, 2*np.pi)
            agents[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    return agents

def update_states(S_agents, I_agents, R_agents, infection_radius, alpha):

    S_positions = np.array(S_agents)
    I_positions = np.array(I_agents)
    R_positions = np.array(R_agents)

    if len(S_agents) > 0 and len(I_agents) > 0:
        diff = S_positions[:, np.newaxis, :] - I_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis = 2)

        # Determine which Susceptible agents are within infection_radius of any Infected agent
        within_infection_radius = np.any(distances < infection_radius, axis=1)

        # Indices of Susceptible agents who are close to at least one Infected agent
        susceptible_close_indices = np.where(within_infection_radius)[0]
        infection_probs = np.random.uniform(0, 1, len(susceptible_close_indices))
        new_infections_indices = susceptible_close_indices[infection_probs >= alpha]

        newly_infected = [S_agents[i] for i in new_infections_indices]
        I_agents.append(newly_infected)
        for index in sorted(new_infections_indices):
            del S_agents[index]
    
    # Recovery
    if len(I_agents) > 0:
        recovery_prob = np.random.uniform(0, 1, len(I_agents))
        new_recovered_indices = np.where[recovery_prob > beta][0]

        newly_recovered = [I_agents[i] for i in new_recovered_indices]
        R_agents.append(newly_recovered)
        #Removing recoverd from infected.
        for index in sorted(new_recovered_indices):
            del I_agents[index]


    #Immune loss
    if len(R_agents) > 0 and gamma > 0:
        immune_probs = np.random.uniform(0, 1, len(R_agents))
        immune_loss_indices = np.where[immune_probs > gamma][0]

        new_susc = [R_agents[i] for i in immune_loss_indices]
        S_agents.append(new_susc)

        for index in sorted(immune_loss_indices):
            del R_agents[index]

    return S_agents, I_agents, R_agents

def simulate(num_steps, city_radius, S0, I0, R0):

    S_agents, I_agents, R_agents = initialize_agents(agents_num, city_radius, S0, I0, R0)

    # Lists to record daily counts
    S_counts = []
    I_counts = []
    R_counts = []

    for t in range(num_steps):

        S_agents = movements(S_agents, sigma, city_radius)
        I_agents = movements(I_agents, sigma, city_radius)
        R_agents = movements(R_agents, sigma, city_radius)

        S_agents, I_agents, R_agents = update_states(S_agents, I_agents, R_agents, infection_radius, alpha)

        S_counts.append(S_agents)
        I_counts.append(I_agents)
        R_counts.append(R_agents)

        # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(S_counts, label='Susceptible', color='blue')
    plt.plot(I_counts, label='Infected', color='red')
    plt.plot(R_counts, label='Recovered', color='green')
    plt.xlabel('Day')
    plt.ylabel('Number of Agents')
    plt.title('SIR Model Simulation without State Encoding in a Circular City')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


simulate(num_steps, city_radius, S0, I0, R0)


