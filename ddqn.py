import numpy as np
import torch
import datetime
from unityagents import UnityEnvironment
from ddqn_agent import MultiAgent, Agent
from collections import deque

env = UnityEnvironment(file_name='app/Reacher.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = MultiAgent(state_size=33, action_size=4, num_agents=20, random_seed=2)

import matplotlib.pyplot as plt
# %matplotlib inline

def ddpg(n_episodes=1000, max_t=1200, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    times = []
    for i_episode in range(1, n_episodes+1):
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        time_a = datetime.datetime.now()
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            
            # print('\rEpisode {}\tScore: {}\tTimestep: {}\tAction: {}\t\t'.format(i_episode, scores.mean(), t, actions[0]), end="")
            
            # need to step for each state
            # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            #     agent.step(state, action, reward, next_state, done)
            agent.step(states, actions, rewards, next_states, dones)
            
            states = next_states
            
            if np.any(dones):
                break
                
        time_b = datetime.datetime.now()
        time = time_b - time_a
        times.append(time_b - time_a)
        time_average = np.mean(times)
        time_remaining = (n_episodes*time_average.total_seconds()-i_episode*time_average.total_seconds())/60/60

        scores_deque.append(scores)

        hours = int(time_remaining)
        minutes = int((time_remaining*60) % 60)
        seconds = int((time_remaining*3600) % 60)

        print('Episode {}\tAverage Score: {:.2f}\tAverage Time Per Episode: {}\tTime to Complete: {:02d}:{:02d}:{:02d}'.format(
            i_episode,
            np.mean(scores_deque),
            time_average,
            hours,
            minutes,
            seconds
        ), end="\n")

        agent.save('checkpoint_actor_ddqn', 'checkpoint_critic_ddqn')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 30:
            print('\nSolved! Episode: {}, Solved after {} episodes! Average score over last 100 episodes: {}'.format(
                i_episode, i_episode - 100, np.mean(scores_deque)
            ))
            
    return scores

# test execution
# scores = ddpg(n_episodes=2, max_t=1200, print_every=100)
scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')
# plt.show()