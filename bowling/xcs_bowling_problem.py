import gym

from xcs.scenarios import Scenario
from xcs.bitstrings import BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder
from xcs.input_encoding.real.center_spread.bitstrings import BitString as BitStringRealEncoded
from xcs.input_encoding.real.scenarios import ExploitTrackingScenarioObserver


class BowlingProblem(Scenario):
    """
    Solves Bowling problem on OpenAI's gym.
    Encodes information as doubles.
    TODO: finish this documentation!
    """

    def __init__(self):
        self.gym_env = gym.make('Bowling-ram-v0')
        self.observation = self.gym_env.reset()
        self.games_to_play = 10
        # a pixel can take values in the (0,255) range, and I will encode the values with 8 bits.
        self.real_translator = EncoderDecoder(min_value=0, max_value=255, encoding_bits=8)

    @property
    def is_dynamic(self):
        return True

    def get_possible_actions(self):
        return list(range(self.gym_env.action_space.n))  # list with elements

    def execute(self, action):
        # print("action => %d " % action)
        self.observation, reward, done, info = self.gym_env.step(action)
        if done:
            self.gym_env.reset()
            self.games_to_play -= 1
        self.gym_env.render()
        # if reward > 0:
        #     print("**************************************** reward = %.2f" % reward)
        return reward

    def sense(self):
        """Senses environment, encodes it as doubles on a bitstring."""
        return BitStringRealEncoded(encoder=self.real_translator, reals=self.observation)

    def more(self):
        return self.games_to_play > 0
        # return not self.done

    def reset(self):
        self.games_to_play = 10
        # print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # self.games_to_play -= 1
        # self.gym_env.reset()


if __name__ == "__main__":
    import logging
    from xcs.algorithms.xcs import XCSAlgorithm
    from xcs.framework import ClassifierSet

    from xcs.scenarios import ScenarioObserver


    import time

    # Setup logging so we can see the test run as it progresses.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # or:
    # logging.root.setLevel(logging.INFO)

    # # Create the scenario instance
    # bowling_problem = BowlingProblem()
    #
    # # Wrap the scenario instance in an observer so progress gets logged,
    # bowling_scenario = ScenarioObserver(bowling_problem)
    # # and pass it on to the test() function.
    # # xcs.test(scenario=bowling_scenario)
    # # or do the whole thing in a detailed way:
    #
    # algorithm = XCSAlgorithm()
    # algorithm.exploration_probability = .1
    # algorithm.do_ga_subsumption = True
    # algorithm.do_action_set_subsumption = False  # TODO: haven't implemented rule subsumption for real representation. Do it!
    # # Create the classifier system from the algorithm.
    # model = ClassifierSet(algorithm, bowling_scenario.get_possible_actions())
    # start_time = time.time()
    # model.run(bowling_scenario, learn=True)
    # end_time = time.time()
    #
    # logger.info('Classifiers:\n\n%s\n', model)
    # logger.info("Total time: %.5f seconds", end_time - start_time)
    #
    # # return (
    # #     scenario.steps,
    # #     scenario.total_reward,
    # #     end_time - start_time,
    # #     model
    # # )

    max_exploit_problems = 10000
    pop_size = 800
    # Create the scenario instance
    bowling_problem = BowlingProblem()

    # Wrap the scenario instance in an observer so progress gets logged,
    # bowling_scenario = ExploitTrackingScenarioObserver(
    #     bowling_problem,
    #     max_exploit_problems=max_exploit_problems,
    #     feedback_dir="/tmp/luis/tests/bowling/%d" % (pop_size))
    bowling_scenario = ScenarioObserver(bowling_problem)
    # and pass it on to the test() function.
    # xcs.test(scenario=bowling_scenario)
    # or do the whole thing in a detailed way:

    algorithm = XCSAlgorithm()
    # parameters as of original paper:
    algorithm.max_population_size = pop_size          # N
    algorithm.learning_rate = .2                # beta
    algorithm.accuracy_coefficient = .1          # alpha # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.error_threshold = 10              # epsilon_0
    #algorithm.accuracy_power = 5                 # nu # TODO: is this 'n'?
    #algorithm.ga_threshold = 12                  # theta_GA
    algorithm.crossover_probability = .8        # chi
    algorithm.mutation_probability = .04         # mu
    #algorithm.deletion_threshold = 20            # theta_del # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.fitness_threshold = .1             # delta # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.subsumption_threshold = 20         # theta_sub # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.initial_prediction = 10        # p_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.initial_error = 0             # epsilon_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.initial_fitness = .01           # F_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    #algorithm.minimum_actions = 2             # theta_mna # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # algorithm.error_threshold = 1.0              # epsilon_0  # TODO: is this s_0 ?
    # we want that, on average, we present exploitation and exploration problems:
    algorithm.exploration_probability = .5       # p_exp

    # these parameters come from page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf:
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = False


    # discount_factor = .71              # gamma
    # fitness_threshold = .1             # delta
    # subsumption_threshold = 20         # theta_sub
    # wildcard_probability = .33         # P_#

    # Create the classifier system from the algorithm.
    model = ClassifierSet(algorithm, bowling_scenario.get_possible_actions())
    start_time = time.time()
    model.run(bowling_scenario, learn=True)
    end_time = time.time()

    # logger.info('Classifiers:\n\n%s\n', model)
    logger.info("Total time: %.5f seconds", end_time - start_time)

    # return (
    #     scenario.steps,
    #     scenario.total_reward,
    #     end_time - start_time,
    #     model
    # )
