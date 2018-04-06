import gym
from gym import spaces
from xcs.scenarios import Scenario
from xcs.bitstrings import BitString

class BowlingProblem(Scenario):

    def __init__(self):
        self.gym_env = gym.make('Bowling-ram-v0')
        self.observation = self.gym_env.reset()
        self.done = False

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return list(range(self.gym_env.action_space.n))  # list with elements

    def execute(self, action):
        self.observation, reward, self.done, info = self.gym_env.step(action)
        self.gym_env.render()
        return reward

    def sense(self):
        # TODO: encode this as a bitstring!
        return BitString('101')
        # return self.observation

    def more(self):
        return not self.done

    def reset(self):
        self.gym_env.reset()


if __name__ == "__main__":
    import logging
    import xcs

    from xcs.scenarios import ScenarioObserver

    # Setup logging so we can see the test run as it progresses.
    logging.root.setLevel(logging.INFO)

    # Create the scenario instance
    problem = BowlingProblem()

    # Wrap the scenario instance in an observer so progress gets logged,
    # and pass it on to the test() function.
    xcs.test(scenario=ScenarioObserver(problem))
