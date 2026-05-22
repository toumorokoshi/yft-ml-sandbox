"""
An example of training a blackjack agent using reinforcement learning.
"""
from gymnasium import gym

class BlackjackAgent:
    def __init__(self):

agent = BlackjackAgent()
# this is the number of times we want the agent to go through a
# training loop.
EPISODE_COUNT = 1000

def train(env: gym.Env):
    env = gym.make("Blackjack-v1")
    for episode in range(EPISODE_COUNT):
