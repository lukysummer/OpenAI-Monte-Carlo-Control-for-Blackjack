import numpy as np
import gym
import sys
from collections import defaultdict

'''
Create an instance of openAI gym's Black Jack environment, which has:
    
        state = tuple(current sum of player's cards, 
                      dealer's face-up card, 
                      whether or not the player has a usable Ace)   # means Ace & 10 card 
'''
env = gym.make('Blakcjack-v0')



def get_probs(Q_state, epsilon, nA):
    '''
    Obtains action probabilities corresponding to epsilon-greedy policy
    ___Argumnets___
        Q_state : (subset of Q) array containing action-value functions for a single state
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
        nA      : # of actions
    '''
    # 1. Initialize w/ probabilities for non-greedy actions:
    probs = np.ones(nA) * epsilon/nA
    # 2. Set the probability for greedy action:
    greedy_action = np.argmax(Q_state)
    probs[greedy_action] = 1 - epsilon + epsilon/nA
    
    return probs



def generate_episode(env, Q, epsilon, nA):
    '''
    Generates an episode using policy = epsilon-greedy(Q)
    ___Arguments___
        env     : openAI gym environment
        Q       : action value function table
        epsilon : p(equiprobable random policy)  &&  1 - epsilon : p(greedy policy)
        nA      : # of actions
    '''   
    episode = []
    state = env.reset()  # Get Initial State (S0)
    
    while True:
        # Generate action with epsilon-greedy policy
        action = np.random.choice(np.arange(2), p = get_probs(Q[state], epsilon, nA)) \
                        if state in Q else env.action_space.sample() 
        next_state, reward, done, info = env.step(action)
        episode.append((state, reward, action))
        state = next_state
        
        if done:
            break
        
    return episode
        
        

def update_Q(env, episode, Q, alpha, gamma):
    '''
    Updates Q using the most recently generated episode
    ___Arguments___
        env     : openAI gym environment
        episode : most recently generated episode
        Q       : action value function table
        alpha   : constant rate for updating action value functions in Q
        gamma   : reward discount rate
    '''
    states, actions, rewards = zip(*episode)
    discounts = [gamma**t for t in range(len(rewards+1))]
    
    for t, state in enumerate(states):
        old_Qsa = Q[state][actions[t]]
        # Return at time step t (Gt):
        G_t = sum(discounts[:-(t+1)] * rewards[t:]) 
        Q[state][actions[t]] = old_Qsa + alpha*(G_t - old_Qsa) 
        
    return Q    
        
        
        
def mc_control(env, num_episodes, alpha, gamma = 1.0, 
               eps_start = 1.0, eps_decay = 0.99999, eps_min = 0.05):
    '''
    Executes Constant-Alpha Monte Carlo Control, using epsilon-greedy policy for each episode 
    ___Arguments___
        env         : openAI gym environment
        num_episode : total # of episodes to iterate over
        alpha       : constant rate for updating action value functions in Q
        gamma       : reward discount rate
        eps_start   : initial epsilon value
        eps_decay   : rate at which epsilon will decay after each episode
        eps_min     : smallest allowable value of epsilon 
                      (epsilon will stop decaying at this value and stay constant afterwards)
    '''
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    
    for i_episode in range(num_episodes):
        
        # 1. Update epsilon:
        epsilon = max(epsilon*eps_decay, eps_min)
        
        # 2. Generate episode:
        episode = generate_episode(env, Q, epsilon, nA)
        
        # 3. Update Q
        Q = update_Q(env, episode, Q, alpha, gamma)

        # 4. Monitor progress:
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}.'.format(i_episode, num_episodes), end = "")
            sys.stdout.flush()
        
    # 5. Construct a greedy Policy using the optimized Q over all iterations
    policy = dict((state, np.argmax(action)) for state, action in Q.items())
    
    return policy, Q        
        


'''Execute Constant-Alpha Monte Carlo Control'''
num_episodes = 500000
alpha = 0.02
policy, Q = mc_control(env, num_episodes, alpha)   



'''Plot results using helper file provided by Udacity Deep Reinforcement Learning Nanodegree'''
from plot_utils import plot_blackjack_values, plot_policy

# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

# plot the policy
plot_policy(policy)
