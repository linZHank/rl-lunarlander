import gym

env = gym.make('LunarLander-v2')
env.reset()
c = 0 # episode length counter
while True:
    env.render()
    a = env.action_space.sample()
    o, r, d, i = env.step(a)
    c += 1
    if d or c>=env.spec.max_episode_steps:
        env.reset()
        c = 0


