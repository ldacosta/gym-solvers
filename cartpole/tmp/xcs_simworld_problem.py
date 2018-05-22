
from random import random
from typing import List, Dict

from xcs.scenarios import Scenario, ScenarioObserver
from xcs.input_encoding.real.center_spread.bitstrings import BitString as BitStringRealEncoded
from cartpole.tmp.sim_world import SimWorld, Direction
from xcs.input_encoding.real.center_spread.util import EncoderDecoder

import logging
from xcs.algorithms.xcs import XCSAlgorithm
from xcs.framework import ClassifierSet

import time
from PIL import Image
import math
import numpy as np
from typing import Tuple

from util.general import get_free_file_name


class SimWorldProblem(Scenario):

    REWARD_ON_SUCCESS = 1
    PUNISHMENT_ON_NON_SUCCESS = -1

    def __init__(self, size: int, alogger: logging.Logger):
        self.logger = alogger
        self.world = SimWorld(size=size, toroidal=True)
        self.world.place_ball()
        self.reset()
        self.real_translators = [
            EncoderDecoder(min_value=0, max_value=self.world.max_distance, encoding_bits=8),
            EncoderDecoder(min_value=-180, max_value=180, encoding_bits=8)
        ]

    def reset(self):
        self.logger.debug("Resetting SimWorldProblem")
        self.world.place_agent(avoid_ball_position=True)
        self.original_dist_ball_to_agent = self.world.dist_agent_to_ball()
        self.steps = 0
        self.episode_done = False
        # TODO: everything that is down here is tmp measures to see if things work.
        self.orig_ball = self.world.ball_pos
        self.orig_agent = self.world.agent_pos
        d_xy = self.world._dist_pt2pt(pt1=self.orig_agent, pt2=self.orig_ball)
        self.min_steps_to_solve = abs(d_xy[0]) + abs(d_xy[1])

    @property
    def is_dynamic(self):
        return True

    def get_possible_actions(self):
        return [Direction.RIGHT, Direction.LEFT, Direction.TOP, Direction.BOTTOM]

    def execute(self, action, **kwargs):
        self.logger.debug("[steps={}] action => {} ".format(self.steps, action.name))
       #  self.logger.debug("ENTERING move_agent ")
        self.world.move_agent(direction=action)
        # self.logger.debug("EXITED move_agent ")
        # self.logger.debug("ENTERING question ->  agent_at_ball_position()")
        b = self.world.agent_at_ball_position()
        # self.logger.debug("EXITED question ->  agent_at_ball_position()")
        if b:
            reward = SimWorldProblem.REWARD_ON_SUCCESS
            self.episode_done = True
        else:
            reward = SimWorldProblem.PUNISHMENT_ON_NON_SUCCESS
            self.episode_done = False
        self.steps += 1
        if self.episode_done and (self.steps < self.min_steps_to_solve):
            self.logger.debug("orig ball = {}, current ball = {}".format(self.orig_ball, self.world.ball_pos))
            self.logger.debug("orig agent = {}, current agent = {}".format(self.orig_agent, self.world.agent_pos))
            self.logger.debug("distance at beginning = %.2f; steps = %d" % (self.min_steps_to_solve, self.steps))
            raise RuntimeError("lalala")
        self.logger.debug("[steps=%d] reward = %.2f, episode done = %s" % (self.steps, reward, self.episode_done))
        return reward

    def sense(self):
        if self.steps > pow(self.world.size, 2):
            self.logger.debug("Something is up: [steps=%d in a world that is %d x %d] ! <==================" % (self.steps, self.world.size, self.world.size))
        self.logger.debug("[steps=%d] sense!" % (self.steps))
        r = BitStringRealEncoded(encoders=self.real_translators,
                             reals=[self.world.dist_agent_to_ball(), self.world.angle_agent_to_ball()])
        self.logger.debug("[steps={}] sense => return {}".format(self.steps, r))
        return r
        # return self.world.agent_sensing()

    def more(self):
        return not self.episode_done


def map2img(a_map: np.ndarray, conversion_dict: Dict[int, Tuple[int, int, int]]):
    def choose_img_size_from_matrix_size(matrix_size: int) -> (int, int):
        mult = ((int(math.floor(255 / matrix_size))))
        return mult, mult * matrix_size

    def map2img_value(value_in_map: int) -> Tuple[int,int,int]:
        return conversion_dict[value_in_map]
        # if value_in_map == -1:
        #     return (0,0,0)
        # elif value_in_map == 1:
        #     return (255,51,255) # RIGHT => fucsia
        # elif value_in_map == 2:
        #     return (51,255,255) # LEFT => cyan
        # elif value_in_map == 3:
        #     return (255,255,51) # TOP => yellow
        # elif value_in_map == 4:
        #     return (255,51,51)  # BOTTOM => red
        # elif value_in_map == 100:
        #     return (255,255,255)
        # else:
        #     RuntimeError("value in map == {} ??" % (value_in_map))

    max_x, max_y = a_map.shape
    assert max_x == max_y
    multiplier, the_size = choose_img_size_from_matrix_size(matrix_size=max_x)
    data = np.zeros((the_size, the_size, 3), dtype=np.uint8)
    for x in range(max_x):
        for y in range(max_y):
            init_x = x * multiplier
            init_y = y * multiplier
            data[init_x:init_x + multiplier, init_y:init_y + multiplier] = map2img_value(a_map[x,y])
    return Image.fromarray(data, 'RGB')

def show_img_from(a_map: np.ndarray):
    the_img = map2img(a_map)
    the_img.save('/tmp/lalala.img')
    the_img = Image.open('/tmp/lalala.img')
    the_img.close()

    # map2img(a_map).show(title="tralala")

def create_img_from_array_and_save(a_map: np.ndarray, where: str):
    the_img = map2img(a_map)
    the_img.save(where)

def save_img_from_map(problem: SimWorldProblem, model: ClassifierSet, where: str):
    NO_ACTION = -100
    BEST = 33
    OK = 34
    BAD = 35
    WORST = 36
    BALL_POS = 100
    #
    world = problem.world
    the_map = np.zeros((world.size, world.size))
    for x in range(world.size):
        for y in range(world.size):
            d = world.dist_to_ball(pt=(x,y))
            a = world.angle_to_ball(pt=(x,y))
            all_deltas = sorted([world.distance_to_ball_if_moved(from_pt=(x, y), direction=dir) - d for dir in Direction])
            #
            situation=BitStringRealEncoded(encoders=problem.real_translators, reals=[d,a])
            match_set = model.match(situation)
            if len(match_set) == 0:
                the_map[x,y] = NO_ACTION
            else:
                new_d = world.distance_to_ball_if_moved(from_pt=(x, y), direction=match_set.best_actions[0])
                delta = new_d - d
                idx = all_deltas.index(delta)
                the_map[x, y] = BEST if idx == 0 else (OK if idx == 1 else (BAD if idx == 2 else WORST))
    the_map[world.ball_pos[0], world.ball_pos[1]] = BALL_POS  # position of the ball is marked in a special way.
    # so this map contains:
    # -1 if no action was selected;
    # 0 if the action selected does not decrease distance to ball;
    # 1 if it does decrease it;
    # 2 if it's the ball position
    conversion_dict = {}
    conversion_dict[NO_ACTION] = [0,0,0]  # black
    conversion_dict[BEST] = [0,64,255]  # very blue.
    conversion_dict[OK] = [230,236,255]  # somewhat blue.
    conversion_dict[BAD] = [255,235,230]  # somewhat red.
    conversion_dict[WORST] = [255,51,0]  # very red.
    conversion_dict[BALL_POS] = [255,255,0] # where the ball is => yellow
    the_img = map2img(the_map, conversion_dict)
    the_img.save(where)
    # return the_map



if __name__ == "__main__":

    from util.general import get_logger
    import pathlib, os
    import pickle

    max_exploit_problems = 500  # 10000
    pop_size = 850  # 20100
    world_size=15
    dir_for_imgs = os.path.join('/tmp/luis/imgs', str(world_size))
    dir_for_imgs = os.path.join(dir_for_imgs, str(pop_size))
    pathlib.Path(dir_for_imgs).mkdir(parents=True, exist_ok=True)
    dir_for_perfs = os.path.join('/tmp/luis/simworld', str(world_size))
    dir_for_perfs = os.path.join(dir_for_perfs, str(pop_size))
    pathlib.Path(dir_for_perfs).mkdir(parents=True, exist_ok=True)

    # Setup logging so we can see the test run as it progresses.
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    common_logger = get_logger(name="common_logger", debug_log_file_name=os.path.join(dir_for_imgs, "common_logger.log"))
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))

    # Create the scenario instance
    gettheball_problem = SimWorldProblem(size=world_size, alogger=common_logger)


    algorithm = XCSAlgorithm()
    # parameters as of original paper:
    algorithm.max_population_size = pop_size          # N
    algorithm.discount_factor = 0.9
    # algorithm.learning_rate = .2                # beta
    # algorithm.accuracy_coefficient = .1          # alpha # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.error_threshold = (SimWorldProblem.REWARD_ON_SUCCESS - SimWorldProblem.PUNISHMENT_ON_NON_SUCCESS) / 100              # epsilon_0: set at about 1% of the reward range.
    # #algorithm.accuracy_power = 5                 # nu # TODO: is this 'n'?
    # #algorithm.ga_threshold = 12                  # theta_GA
    algorithm.crossover_probability = 0.8        # chi
    algorithm.mutation_probability = .04         # mu
    algorithm.deletion_threshold = 10            # theta_del: the higher, the linger we give to new rules to demonstrate their value.
    # #algorithm.fitness_threshold = .1             # delta # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.subsumption_threshold = 20         # theta_sub # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_prediction = 10        # p_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_error = 0             # epsilon_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # #algorithm.initial_fitness = .01           # F_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.minimum_actions = 2             # theta_mna # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    # # algorithm.error_threshold = 1.0              # epsilon_0  # TODO: is this s_0 ?
    # # we want that, on average, we present exploitation and exploration problems:
    algorithm.exploration_probability = .35       # p_exp
    #
    # # these parameters come from page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf:
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True # False


    # discount_factor = .71              # gamma
    # fitness_threshold = .1             # delta
    # subsumption_threshold = 20         # theta_sub
    # wildcard_probability = .33         # P_#




    # Create the classifier system from the algorithm.
    model = ClassifierSet(algorithm, gettheball_problem.get_possible_actions())
    start_time = time.time()
    episodes_between_feedback = 20
    rewards = np.empty(episodes_between_feedback)
    num_exploit_episodes = 0
    hist_reward = {}  # each entry contains the (avg, std-dev) value of rewards obtained the last 'episodes_between_feedback' episodes.
    I_have_seen_enough = False
    for episode in range(1, 100000):
        if I_have_seen_enough:
            break  # horrible, but whatev'
        # gettheball_problem.world.place_ball(pos=(1,1))
        exploit_episode = (episode > 10) and (episode % 2 == 0)
        model.algorithm.exploration_probability = 0.0 if exploit_episode else 0.35  # p_exp
        model.run(gettheball_problem, learn=True)
        if episode % 10 == 0:
            base_filename = "optimal_moves_%d" % (episode)
            full_file_name = os.path.join(dir_for_imgs, base_filename + ".jpg")



            save_img_from_map(gettheball_problem, model, where=full_file_name)


#             create_img_from_array_and_save(save_img_from_map(problem = gettheball_problem, model = model), where=full_file_name)
            common_logger.info("********************************************************************** saved visual of actions in %s" % (full_file_name))
            MAX_RULES = 5
            common_logger.info("Episode %d ----------- MAX %d RULES WITH GOOD FITNESS AND LOTS OF EXPERIENCE" % (episode, MAX_RULES))
            num_rules = 0
            for rule in model:
                if rule.fitness > .5 and rule.experience >= 10:
                    common_logger.info("{} => {} (fitness: {}".format(rule.condition, rule.action.name, rule.fitness))
                    num_rules += 1
                    if num_rules == MAX_RULES:
                        break
            common_logger.info("-------- END Episode %d ----------- MAX %d RULES WITH GOOD FITNESS AND LOTS OF EXPERIENCE" % (episode, MAX_RULES))

        if exploit_episode:
            idx_rewards = num_exploit_episodes % episodes_between_feedback
            rewards[idx_rewards] = gettheball_problem.steps / gettheball_problem.min_steps_to_solve
            if num_exploit_episodes > 0 and (num_exploit_episodes % episodes_between_feedback == 0):
                common_logger.info("============> Episode %d (EXPLOIT episode %d): mean performance (last %d exploit episodes) = %.2f +/- %.2f" %
                      (episode, num_exploit_episodes, episodes_between_feedback, np.mean(rewards), np.std(rewards)))
                hist_reward[num_exploit_episodes] = np.mean(rewards), np.std(rewards)
            elif random() < .2:
                common_logger.info("Episode %d (EXPLOIT): steps = %d, original distance = %.2f, performance = %.2f" %
                      (episode, gettheball_problem.steps, gettheball_problem.min_steps_to_solve,
                       rewards[idx_rewards]))

        # if (not exploit_episode) and (random() < (.2 if num_exploit_episodes == 0 else .1)):
        #     common_logger.info("Reward for (non-exploit) episode %d  = steps/orig_dist = %d/%.2f = %.2f " %
        #                  (episode,
        #                   gettheball_problem.steps,
        #                   gettheball_problem.min_steps_to_solve,
        #                   gettheball_problem.steps / gettheball_problem.min_steps_to_solve))
        gettheball_problem.reset()
        num_exploit_episodes += 1 if exploit_episode else 0
        I_have_seen_enough = num_exploit_episodes > max_exploit_problems
    end_time = time.time()

    # persist results
    l = []
    for exploit_episode, (mean_perf, std_dev_perf) in hist_reward.items():
        l.append((exploit_episode, mean_perf, std_dev_perf))

    exploit_success_file = get_free_file_name(a_dir=dir_for_perfs, root="exploit_successes", ext="txt")
    with open(exploit_success_file, 'wb') as fp:
        pickle.dump(hist_reward, fp)
        # pickle.dump(l, fp)
    common_logger.info("Successes on 'exploit' problems saved on file '%s'" % (exploit_success_file))

    # logger.info('Classifiers:\n\n%s\n', model)
    common_logger.info("Total time: %.5f seconds", end_time - start_time)

    # return (
    #     scenario.steps,
    #     scenario.total_reward,
    #     end_time - start_time,
    #     model
    # )
