import gym
import math

from typing import List
import numpy as np

from xcs.scenarios import Scenario, ScenarioObserver
from xcs.bitstrings import BitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder
from xcs.input_encoding.real.center_spread.bitstrings import BitString as BitStringRealEncoded


class CartpoleProblem(Scenario):
    """
    Solves Bowling problem on OpenAI's gym.
    Encodes information as doubles.
    TODO: finish this documentation!
    """
    MIN_VALUE = -5
    MAX_VALUE = 5

    def __init__(self):
        self.gym_env = gym.make('CartPole-v0') # TODO: ('CartPole-v0') ?
        self.observation = self.gym_env.reset()
        self.episode_done = False
        self.reward_on_episode = 0

        highs = self.gym_env.observation_space.high
        lows = self.gym_env.observation_space.low

        if not isinstance(highs, np.ndarray):
            highs = [highs]
            lows = [lows]
        print(highs)
        print(lows)
        self.the_min = -1
        self.the_max = 1
        self.real_translators = [EncoderDecoder(min_value=max(CartpoleProblem.MIN_VALUE, low), max_value=min(CartpoleProblem.MAX_VALUE, high), encoding_bits=4)
                                 for low, high in zip(lows, highs)]

    @property
    def is_dynamic(self):
        return True

    def get_possible_actions(self):
        return list(range(self.gym_env.action_space.n))  # list with elements

    def execute(self, action, **kwargs):
        # print("action => %d " % action)
        self.observation, reward, self.episode_done, info = self.gym_env.step(action)
        self.gym_env.render()
        # print("reward => %d" % (reward))
        self.reward_on_episode += 1  # TODO: just something to keep track outside
        if self.episode_done:
            return 0
        else:
            return reward



        # self.the_min = min(self.the_min, min(self.observation))
        # self.the_max = max(self.the_max, max(self.observation))
        # if not all(map(lambda v: (v > CartpoleProblem.MIN_VALUE) and (v < CartpoleProblem.MAX_VALUE), self.observation)):
        #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH!")
        #     print(self.observation)
        #     raise RuntimeError("captain underpants!!!!!")
        # # print(self.observation)
        self.gym_env.render()
        # # r_d = (-abs(self.observation[0]) / self.real_translators[0].extremes[1]) + 1 # for distance
        # # r_a = (-abs(self.observation[2]) / self.real_translators[2].extremes[1]) + 1 # for angle
        # r_d = (-abs(self.observation[0]) / 2.4) + 1 # for distance
        # r_a = (-abs(self.observation[2]) / 0.21) + 1 # for angle
        # r = r_d * r_a  # (r_d + r_a) / 2
        # return r
        self.reward_on_episode += reward
        if self.episode_done:
            r = self.reward_on_episode - 195
            print("reward => %d" % (r))
            return r
        else:
            return 0

    def sense(self):
        """Senses environment, encodes it as doubles on a bitstring."""
        encoding_bits = 2
        b = BitString('')
        for value, encoder in zip(self.observation, self.real_translators):
            width = (encoder.extremes[1] - encoder.extremes[0]) / 4
            if value <= encoder.extremes[0] + width:
                as_bitstring = ('{0:0%db}' % encoding_bits).format(0)
            elif value <= encoder.extremes[0] + 2 * width:
                as_bitstring = ('{0:0%db}' % encoding_bits).format(1)
            elif value <= encoder.extremes[0] + 3 * width:
                as_bitstring = ('{0:0%db}' % encoding_bits).format(2)
            else:
                as_bitstring = ('{0:0%db}' % encoding_bits).format(3)
            b += as_bitstring
        return b
        # return BitStringRealEncoded(encoders=self.real_translators, reals=self.observation)

    def more(self):
        return not self.episode_done

    def reset(self):
        self.reward_on_episode = 0
        self.observation = self.gym_env.reset()
        self.episode_done = False


class CartpoleScenarioObserver(ScenarioObserver):

    def execute(self, action, **kwargs):
        reward = ScenarioObserver.execute(self, action, **kwargs)

        if self.wrapped.episode_done:
            self.reset()
        return reward

    def reset(self):
        ScenarioObserver.reset(self)
        self.logger.info('reset ===> Reward on episode = %d', self.total_reward)
        # self.logger.info('reset ===> Average reward per step: %.2f/%d = %.5f', self.total_reward, self.steps, self.total_reward / self.steps)
        self.total_reward = 0
        self.steps = 0

# class CartpoleScenarioObserver(ScenarioObserver):
#
#     def __init__(self, wrapped, feedback_dir: str):
#         # Ensure that the wrapped object implements the same interface
#         assert isinstance(wrapped, Scenario)
#
#         ScenarioObserver.__init__(self, wrapped)
#         self.latest_feedback_time = None  # when was the last time I gave feedback to the user
#         self.max_exploit_problems = max_exploit_problems
#         self.feedback_dir = feedback_dir
#         pathlib.Path(self.feedback_dir).mkdir(parents=True, exist_ok=True)
#         self.logger.info("Will leave trace of this run on directory '%s'" % (self.feedback_dir))
#         # let's keep 'number of success(es) on "exploit" problems' as a measure of performance
#         self.window_length = 50  # in number of steps. 50 == taken in https://pdfs.semanticscholar.org/6777/624d3a7230742d1b8f6bc5f8a43d6daf065d.pdf?_ga=2.120249934.77113088.1523040926-378861627.1523040926
#         self.latest_exploit_rewards = queue.Queue(maxsize=self.window_length)
#         self.num_latest_success = 0  # number of successes in current window
#         self.num_exploits = 0  # how many 'exploit'ation problems have I seen
#         self.success_per_period = []  # history of successes per window.
#         self.prediction_error_per_period = []  # history of errors per window.
#         self.sum_predicted_error = 0  # sum of errors in current window
#
#     def _get_free_file_name(self, root: str, ext: str) -> str:
#         fcounter = 1
#         fname = os.path.join(self.feedback_dir, "%s_%d.%s" % (root, fcounter, ext))
#         while os.path.isfile(fname):
#             fcounter += 1
#             fname = os.path.join(self.feedback_dir, "%s_%d.%s" % (root, fcounter, ext))
#         return fname
#
#     @property
#     def exploit_successes_on_window(self) -> Tuple[int, float]:
#         """Returns exploit successes on window, as an absolute value and as a proportion (in [0,1])"""
#         if self.steps == 0:
#             return (0,0)
#         return self.num_latest_success, \
#                self.num_latest_success / (self.steps if self.steps < self.window_length else self.window_length)
#
#     def execute(self, action, **kwargs):
#         self.logger.debug('Executing action: %s', action)
#         if 'is_exploit' not in kwargs:
#             raise RuntimeError("execute: need to know if this action came from exploitation or exploration")
#         if 'predicted_reward' not in kwargs:
#             raise RuntimeError("execute: need to know the prediction made by this action")
#         is_exploit = kwargs['is_exploit']
#         predicted_reward = kwargs['predicted_reward']
#         reward = self.wrapped.execute(action)
#         if reward is not None:
#             self.total_reward += reward
#             if is_exploit:
#                 self.num_exploits += 1
#                 if self.latest_exploit_rewards.full():
#                     # dump this value
#                     self.success_per_period.append(self.num_latest_success)
#                     self.prediction_error_per_period.append(self.sum_predicted_error)
#                     # update statistics
#                     old_reward, old_predicted_reward_error = self.latest_exploit_rewards.get()
#                     self.num_latest_success -= 1 if old_reward > 0 else 0
#                     self.sum_predicted_error -= old_predicted_reward_error
#                 error_on_prediction = abs(predicted_reward - reward)
#                 self.latest_exploit_rewards.put((reward, error_on_prediction))
#                 self.sum_predicted_error += error_on_prediction
#                 self.num_latest_success += 1 if reward > 0 else 0
#         self.steps += 1
#
#         self.logger.debug('Reward received on this step: %.5f',
#                           reward or 0)
#         self.logger.debug('Average reward per step: %.5f',
#                           self.total_reward / self.steps)
#         succ_abs, succ_perc = self.exploit_successes_on_window
#         self.logger.debug('[exploit trials so far: %d] Num successes on latest tries: %d (perc success of %.5f)',
#                           self.num_exploits, succ_abs, succ_perc)
#
#         return reward
#
#     def more(self):
#         """Return a Boolean indicating whether additional actions may be
#         executed, per the reward program.
#
#         Usage:
#             while scenario.more():
#                 situation = scenario.sense()
#                 selected_action = choice(possible_actions)
#                 reward = scenario.execute(selected_action)
#
#         Arguments: None
#         Return:
#             A bool indicating whether additional situations remain in the
#             current run.
#         """
#         more = self.wrapped.more() or (self.num_exploits <= self.max_exploit_problems)
#         current_time = time.time()
#         if self.latest_feedback_time is None or (current_time - self.latest_feedback_time >= 5):  # seconds between feedback
#             self.latest_feedback_time = current_time
#             self.logger.info('Steps completed: %d', self.steps)
#             self.logger.info('Average reward per step: %.5f',
#                              self.total_reward / (self.steps or 1))
#             succ_abs, succ_perc = self.exploit_successes_on_window
#             self.logger.info('[exploit trials so far: %d] Num successes on latest tries: %d (perc success of %.5f)',
#                               self.num_exploits, succ_abs, succ_perc)
#         if not more:
#             self.logger.info('Run completed.')
#             self.logger.info('Total steps: %d', self.steps)
#             self.logger.info('Total reward received: %.5f',
#                              self.total_reward)
#             self.logger.info('Average reward per step: %.5f',
#                              self.total_reward / (self.steps or 1))
#             # save successes to disk:
#             import pickle
#
#             exploit_success_file = self._get_free_file_name(root="exploit_successes", ext="txt")
#             with open(exploit_success_file, 'wb') as fp:
#                 pickle.dump(self.success_per_period, fp)
#             self.logger.info("Successes on 'exploit' problems saved on file '%s'" % (exploit_success_file))
#             error_on_prediction_file = self._get_free_file_name(root="error_on_prediction", ext="txt")
#             with open(error_on_prediction_file, 'wb') as fp:
#                 pickle.dump(self.prediction_error_per_period, fp)
#             self.logger.info("Errors on predicted rewards saved on file '%s'" % (error_on_prediction_file))
#         return more

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


    max_exploit_problems = 10000
    pop_size = 1620  # 20100
    # Create the scenario instance
    cartpole_problem = CartpoleProblem()


    algorithm = XCSAlgorithm()
    # parameters as of original paper:
    algorithm.max_population_size = pop_size          # N
    algorithm.discount_factor = 0.95
    algorithm.learning_rate = .2                # beta
    # algorithm.accuracy_coefficient = .1          # alpha # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.error_threshold = 10              # epsilon_0
    # #algorithm.accuracy_power = 5                 # nu # TODO: is this 'n'?
    algorithm.ga_threshold = 25                  # theta_GA
    algorithm.crossover_probability = 1        # chi
    algorithm.mutation_probability = .01         # mu
    # algorithm.deletion_threshold = 10            # theta_del # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.fitness_threshold = .1             # delta # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.subsumption_threshold = 20         # theta_sub # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_prediction = 10        # p_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_error = 0             # epsilon_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_fitness = .01           # F_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.minimum_actions = 2             # theta_mna # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # # algorithm.error_threshold = 1.0              # epsilon_0  # TODO: is this s_0 ?
    # # we want that, on average, we present exploitation and exploration problems:
    # algorithm.exploration_probability = .2       # p_exp
    #
    # # these parameters come from page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf:
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = False


    # discount_factor = .71              # gamma
    # fitness_threshold = .1             # delta
    # subsumption_threshold = 20         # theta_sub
    # wildcard_probability = .33         # P_#

    # Create the classifier system from the algorithm.
    model = ClassifierSet(algorithm, cartpole_problem.get_possible_actions())
    start_time = time.time()
    episodes_between_feedback = 20
    rewards = np.empty(episodes_between_feedback)
    for episode in range(1, 100000):
        # Wrap the scenario instance in an observer so progress gets logged
        # cartpole_scenario = ScenarioObserver(cartpole_problem)
        # model.run(cartpole_scenario, learn=True)
        model.run(cartpole_problem, learn=True)
        if episode % 100 == 0:
            print(model)
        # print("between %.2f and %.2f" % (cartpole_problem.the_min, cartpole_problem.the_max))
        rewards[(episode - 1) % episodes_between_feedback] = cartpole_problem.reward_on_episode
        if episode % episodes_between_feedback == 0:
            logging.info("Episode %d; mean reward (last %d episodes) = %.2f" % (episode, episodes_between_feedback, np.mean(rewards)))
        cartpole_problem.reset()
    end_time = time.time()

    # logger.info('Classifiers:\n\n%s\n', model)
    logger.info("Total time: %.5f seconds", end_time - start_time)
    cartpole_problem.gym_env.close()

    # return (
    #     scenario.steps,
    #     scenario.total_reward,
    #     end_time - start_time,
    #     model
    # )
