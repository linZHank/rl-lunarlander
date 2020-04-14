import gym

env = gym.make('LunarLander-v2')

for _ in range(10):
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample() # random action
        obs, rew, done, info = env.step(action)
        print(obs, rew, done, info)
        if done:
            break
