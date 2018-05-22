from random import random
from typing import List, Dict, Optional
import pandas as pd
import math
import time
import logging
import itertools

from xcs.scenarios import Scenario, ScenarioObserver
from xcs import XCSAlgorithm, ClassifierSet
from xcs.input_encoding.real.center_spread.bitstrings import BitString as BitStringRealEncoded
from xcs.input_encoding.real.center_spread.util import EncoderDecoder


class TitanicProblem(Scenario):

    REWARD_ON_SUCCESS = 1
    PUNISHMENT_ON_NON_SUCCESS = 0

    def __init__(self, path_to_data: str, training_cycles: int, alogger: logging.Logger):
        self.possible_actions = (True, False)
        self.logger = alogger
        self.data = pd.read_csv(path_to_data)
        self.correct_response = None
        self.remaining_cycles = training_cycles
        self.real_translators = [
            EncoderDecoder(min_value=1, max_value=3, encoding_bits=2),  # passenger class
            EncoderDecoder(min_value=-0, max_value=1, encoding_bits=1),  # sex
            EncoderDecoder(min_value=-0, max_value=100, encoding_bits=6),  # age
            EncoderDecoder(min_value=-0, max_value=10, encoding_bits=3),  # siblings or spouses
            EncoderDecoder(min_value=-0, max_value=10, encoding_bits=3),  # parents or children
            EncoderDecoder(min_value=-0, max_value=520, encoding_bits=9),  # fare
            EncoderDecoder(min_value=-0, max_value=2, encoding_bits=2),  # port of embarcation
        ]

    def _encode_passenger(self, p_class: int, sex: str, age: int, siblings_or_spouses: int, parents_or_children: int, fare: float, embarked: str) -> List[BitStringRealEncoded]:
        """
        See https://www.kaggle.com/c/titanic/data
        :param p_class:
        :param sex:
        :param age:
        :param siblings_or_spouses:
        :param parents_or_children:
        :param fare:
        :param embarked: C = Cherbourg, Q = Queenstown, S = Southampton
        :return:
        """
        options_on_features = []
        # p_class
        if (p_class == 1) or (p_class == 2) or (p_class == 3):
            options_on_features = options_on_features + [[p_class]]
        else:
            self.logger.debug("Class '%d' is not valid. Expanding to all possibilities." % (p_class))
            options_on_features = options_on_features + [[1, 2, 3]]
        # sex
        sex_to_upper = sex.upper()
        if sex_to_upper == 'MALE':
            options_on_features = options_on_features + [[0]]
        elif sex_to_upper == 'FEMALE':
            options_on_features = options_on_features + [[1]]
        else:
            self.logger.debug("Sex '%s' is not valid. Expanding to all possibilities." % (sex))
            options_on_features = options_on_features + [[0, 1]]
        # age
        if not math.isnan(age):
            options_on_features = options_on_features + [[age]]
        else:
            self.logger.debug("Age must be defined. Expanding to multiple possibilities.")
            options_on_features = options_on_features + [[0, 5, 10, 20, 30, 40, 50, 60, 70, 80]]
        # siblings_or_spouses
        options_on_features = options_on_features + [[siblings_or_spouses]]
        # parents_or_children
        options_on_features = options_on_features + [[parents_or_children]]
        # fare
        if not math.isnan(fare):
            options_on_features = options_on_features + [[fare]]
        else:
            self.logger.debug("Fare must be defined")
            options_on_features = options_on_features + [list(range(1,50,2)) + list(range(51,110,10)) + [150, 200, 300]]
        # embarcation port
        if embarked == 'C':
            options_on_features = options_on_features + [[0]]
        elif embarked == 'Q':
            options_on_features = options_on_features + [[1]]
        elif embarked == 'S':
            options_on_features = options_on_features + [[2]]
        else:
            self.logger.debug("Embarked == '%s' is not valid" % (embarked))
            options_on_features = options_on_features + [[0, 1, 2]]
        # ok. Now to generate ALL options:
        all_options = list(itertools.product(*options_on_features))
        r = list(map(lambda opt_list: BitStringRealEncoded(encoders=self.real_translators, reals=opt_list), all_options))
        return r

        #
        # if (p_class != 1) and (p_class != 2) and (p_class != 3):
        #     self.logger.debug("Class '%d' is not valid" % (p_class))
        #     return None
        # sex_to_upper = sex.upper()
        # if sex_to_upper == 'MALE':
        #     sex_as_int = 0
        # elif sex_to_upper == 'FEMALE':
        #     sex_as_int = 1
        # else:
        #     self.logger.debug("Sex '%s' is not valid" % (sex))
        #     return None
        # if math.isnan(age):
        #     self.logger.debug("Age must be defined")
        #     return None
        # if math.isnan(fare):
        #     self.logger.debug("Fare must be defined")
        #     return None
        # if embarked == 'C':
        #     embarked_as_int = 0
        # elif embarked == 'Q':
        #     embarked_as_int = 1
        # elif embarked == 'S':
        #     embarked_as_int = 2
        # else:
        #     self.logger.debug("Embarked == '%s' is not valid" % (embarked))
        #     return None
        # try:
        #     r = BitStringRealEncoded(
        #         encoders=self.real_translators,
        #         reals=[p_class, sex_as_int, age, siblings_or_spouses, parents_or_children, fare, embarked_as_int])
        # except AssertionError as ae:
        #     print(str(ae))
        #     raise ae
        # return r

    @property
    def is_dynamic(self):
        """A Boolean value indicating whether earlier actions from the same
        run can affect the rewards or outcomes of later actions."""
        return False

    def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be
        executed within the environment.

        Usage:
            possible_actions = scenario.get_possible_actions()

        Arguments: None
        Return:
            A sequence containing the possible actions which can be
            executed within this scenario.
        """
        return self.possible_actions

    def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        self.remaining_cycles = self.initial_training_cycles

    def _encode_row(self, entry) -> List[BitStringRealEncoded]:
        return self._encode_passenger(
            p_class=entry["Pclass"],
            sex=entry["Sex"],
            age=entry["Age"],
            siblings_or_spouses=entry["SibSp"],
            parents_or_children=entry["Parch"],
            fare=entry["Fare"],
            embarked=entry["Embarked"])


    def sense(self):
        """Return a situation, encoded as a bit string, which represents
        the observable state of the environment.

        Usage:
            situation = scenario.sense()
            assert isinstance(situation, BitString)

        Arguments: None
        Return:
            The current situation.
        """
        a_row = self.data.sample(1).iloc[0]
        encoding_options = self._encode_row(a_row)
        self.correct_response = bool(a_row["Survived"])
        while len(encoding_options) > 1:  # data is missing or error occurred.
            a_row = self.data.sample(1).iloc[0]
            encoding_options = self._encode_row(a_row)
            self.correct_response = bool(a_row["Survived"])
        self.remaining_cycles -= 1
        s = encoding_options[0]
        return s

    def execute(self, action, **kwargs):
        """Execute the indicated action within the environment and
        return the resulting immediate reward dictated by the reward
        program.

        Usage:
            immediate_reward = scenario.execute(selected_action)

        Arguments:
            action: The action to be executed within the current situation.
        Return:
            A float, the reward received for the action that was executed,
            or None if no reward is offered.
        """

        assert action in self.possible_actions

        return TitanicProblem.REWARD_ON_SUCCESS if action == self.correct_response else TitanicProblem.PUNISHMENT_ON_NON_SUCCESS

    def more(self):
        """Return a Boolean indicating whether additional actions may be
        executed, per the reward program.

        Usage:
            while scenario.more():
                situation = scenario.sense()
                selected_action = choice(possible_actions)
                reward = scenario.execute(selected_action)

        Arguments: None
        Return:
            A bool indicating whether additional situations remain in the
            current run.
        """
        return int(self.remaining_cycles > 0)


if __name__ == "__main__":
    from util.general import get_logger
    import pathlib, os
    from xcs.input_encoding.real.scenarios import ExploitTrackingScenarioObserver

    max_exploit_problems = 4000
    pop_size = 500
    # Setup logging so we can see the test run as it progresses.
    dir_for_logging = os.path.join(os.path.join('/tmp/luis/titanic/', str(pop_size)), str(max_exploit_problems))
    pathlib.Path(dir_for_logging).mkdir(parents=True, exist_ok=True)
    common_logger = get_logger(name="common_logger", debug_log_file_name=os.path.join(dir_for_logging, "common_logger.log"))
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))

    # Create the scenario instance
    titanic_problem = TitanicProblem(path_to_data="/Users/dacostlu/.kaggle/competitions/titanic/train.csv", training_cycles=2 * max_exploit_problems, alogger=common_logger)

    # Wrap the scenario instance in an observer so progress gets logged,
    titanic_scenario = ExploitTrackingScenarioObserver(
        titanic_problem,
        alogger=titanic_problem.logger,
        max_exploit_problems=max_exploit_problems,
        feedback_dir=dir_for_logging)
    # titanic_scenario = ScenarioObserver(titanic_problem)
    # and pass it on to the test() function.
    # xcs.test(scenario=bowling_scenario)
    # or do the whole thing in a detailed way:

    algorithm = XCSAlgorithm()
    # parameters as of original paper:
    algorithm.max_population_size = pop_size          # N
    algorithm.learning_rate = .8                # beta
    algorithm.accuracy_coefficient = .1          # alpha
    algorithm.error_threshold = (TitanicProblem.REWARD_ON_SUCCESS - TitanicProblem.PUNISHMENT_ON_NON_SUCCESS) / 100              # epsilon_0: set at about 1% of the reward range.
    algorithm.accuracy_power = .5                 # nu # TODO: is this 'n'?
    algorithm.ga_threshold = 12                  # theta_GA
    algorithm.crossover_probability = .8        # chi
    algorithm.mutation_probability = .04         # mu
    algorithm.deletion_threshold = 20            # theta_del # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.fitness_threshold = .1             # delta # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.subsumption_threshold = 20         # theta_sub # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_prediction = 0        # p_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_error = 0.5             # epsilon_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.initial_fitness = .01           # F_I # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
    algorithm.minimum_actions = 2             # theta_mna # page 5 of http://eprints.uwe.ac.uk/5887/1/106365603322365315.pdf
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
    model = ClassifierSet(algorithm, titanic_scenario.get_possible_actions())
    start_time = time.time()
    model.run(titanic_scenario, learn=True)
    end_time = time.time()

    # logger.info('Classifiers:\n\n%s\n', model)
    common_logger.info("Total time: %.5f seconds", end_time - start_time)

    # at the end, let's generate the file to send to Kaggle:

    test_data = pd.read_csv("/Users/dacostlu/.kaggle/competitions/titanic/test.csv")
    L = []
    for index, row in test_data.iterrows():
        pass_encs = titanic_problem._encode_row(row)
        all_responses = list(map(lambda pass_enc: model.match(situation=pass_enc).best_actions[0], pass_encs))
        # take the value more often mentioned:
        pred = max(set(all_responses), key=all_responses.count)
        L = L + [(row['PassengerId'], 1 if pred else 0)]
        # print("%d, %d" % (row['PassengerId'], pred))

    df = pd.DataFrame(data=L, columns=['PassengerId', 'Survived'])
    # print(df)
    where_to_save = os.path.join(dir_for_logging, 'test_results_3.csv')
    df.to_csv(where_to_save, index=False)
    print("Results saved on '%s'" % (where_to_save))

    # return (
    #     scenario.steps,
    #     scenario.total_reward,
    #     end_time - start_time,
    #     model
    # )

