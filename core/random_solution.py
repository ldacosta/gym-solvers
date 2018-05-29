import gym


def cycle_with_random(env_name: str, num_episodes: int):
    """
    Random solution for any existing environment.

    :param env_name:
    :param num_episodes:
    :return:
    """

    print("Making environment '%s'..." % (env_name))
    env = gym.make(env_name)
    print("[%s] action space: " % (env_name))
    print(env.action_space)
    print("[%s] observation space:" % (env_name))
    print(env.observation_space)
    # print("observation range: ")
    # print("\t high: ")
    # print(env.observation_space.high)
    # print("\t low: ")
    # print(env.observation_space.low)
    for episode in range(num_episodes):
        print("Iteration %d" % (episode + 1))
        observation = env.reset()
        done = False
        while not done:
            env.render()
            print(observation)
            observation, reward, done, info = env.step(env.action_space.sample())
            print("Observation ==> ")
            print(observation)

    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())  # take a random action
