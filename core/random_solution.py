import gym


def cycle_with_random(env_name: str, num_episodes: int):
    """
    Random solution for any exisitng environment.

    :param env_name:
    :param num_episodes:
    :return:
    """

    print("Making environment '%s'..." % (env_name))
    env = gym.make(env_name)
    print("[%s] action space: ")
    print(env.action_space)
    print("[%s] observation space:")
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    for episode in range(num_episodes):
        print("Iteration %d" % (episode + 1))
        observation = env.reset()
        done = False
        while not done:
            env.render()
            observation, reward, done, info = env.step(env.action_space.sample())
            # print(observation)

    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())  # take a random action
