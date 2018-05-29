import numpy as np
import gym
import tensorflow as tf

class FrisbeeSearcher(object):

    def __init__(self, env):
        self.Q = np.zeros(shape=[env.observation_space.n, env.action_space.n])  # table with Q values
        self.env = env

    def train(self, **params):
        """

        :param num_episodes:
        :param gamma: discount factor for reward
        :param exploration_value: value between 0 and 1 that drives the random exploration.
        :param learning_rate:
        :return:
        """

        num_episodes = params['num_episodes']
        gamma = params.get('gamma', 0.85)
        exploration_value = params.get('exploration_value', 0.1)
        learning_rate = params.get('learning_rate', 0.85)
        stop_on_perc_successes = params.get('stop_on_perc_successes', 0.75)
        stop_rate_of_success = params.get('stop_rate_of_success', 3/5)
        stop_length_of_checking = params.get('stop_length_of_checking', 10)

        successes_history = []
        check_stop_each = 100  # frequency of checking for success
        episode = 0
        consecutive_checks = []
        for episode in range(num_episodes):
            print("Episode %d" % (episode + 1))
            old_state = self.env.reset()
            done = False
            while not done:
                self.env.render()
                # the action to take is the one that gives us maximum value;
                # we add some noise to this choice to allow for exploration:
                action = np.argmax(self.Q[old_state, :] + np.random.normal(0.0, exploration_value, size=self.env.action_space.n))
                new_state, reward, done, info = self.env.step(action)
                # what is the expected long-term reward of this combination (action, state)?
                expected_q_value = reward + gamma * max(self.Q[new_state, :])
                delta_q = learning_rate * (expected_q_value - self.Q[old_state, action])
                self.Q[old_state, action] += delta_q
                old_state = new_state

            if len(consecutive_checks) > 0 or episode % check_stop_each == 0:
                try_this_many=50
                num_successes = self.count_successes(num_tries=try_this_many)
                rate_of_success = num_successes / try_this_many
                successes_history = successes_history + [(episode, rate_of_success)]
                consecutive_checks = consecutive_checks + [rate_of_success]
                if len(consecutive_checks) == stop_length_of_checking:
                    cum_rate_of_success = len([x for x in consecutive_checks if x >= stop_on_perc_successes]) / stop_length_of_checking
                    if cum_rate_of_success >= stop_rate_of_success:
                        break
                    else:
                        consecutive_checks = []

        print("Finished in %d episodes" % (episode))
        print(self.Q)
        return successes_history

    def count_successes(self, num_tries: int) -> int:
        """Counts."""
        num_successes = 0
        for t in range(num_tries):
            state = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = np.argmax(self.Q[state, :])
                state, reward, done, info = self.env.step(action)
            # this was a success if at the end I got a reward of 1:
            num_successes += (1 if reward == 1 else 0)
        return num_successes

class DeepFrisbeeSearcher(object):
    """ :-) """

    def __init__(self, env):
        self.env = env

    def train(self, num_episodes: int, gamma: float):
        tf.reset_default_graph()

        # definition of network
        inputs = tf.placeholder(shape=[1,16], dtype=tf.float32, name="inputs")
        W = tf.Variable(tf.random_uniform(shape=[16,4], minval=0, maxval=0.01, name="W"))
        Qout = tf.matmul(inputs, W)
        predict = tf.argmax(Qout, axis=1)

        # calculation of loss (square of errors)
        expectedQ = tf.placeholder(shape=[1,4], dtype=tf.float32, name="expectedQ")
        loss = tf.reduce_sum(tf.square(expectedQ - Qout))
        trainer = tf.train.AdamOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(loss)

        # train!
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for episode in range(num_episodes):
                print("Episode %d" % (episode + 1))
                old_state = self.env.reset()
                done = False
                while not done:
                    self.env.render()
                    # one-hot encoding of this state is a [1,16] array:
                    one_hot_input = np.identity(16)[old_state:old_state + 1]
                    # the action to take is the one that gives us maximum value;
                    # we add some noise to this choice to allow for exploration:
                    action_arr, targetQ = sess.run([predict, Qout], feed_dict={inputs: one_hot_input})
                    action = action_arr[0]
                    new_state, reward, done, info = self.env.step(action)
                    # what is the expected long-term reward of this combination (action, state)?
                    # We first have to see what are the Q-valeus associated with _next_ state:
                    new_one_hot_input = np.identity(16)[new_state:new_state + 1]
                    max_nextQ = np.max(sess.run(Qout, feed_dict={inputs: new_one_hot_input}))
                    targetQ[0, action] = reward + gamma * max_nextQ
                    # ok, NOW train:
                    _, W1 = sess.run([updateModel, W], feed_dict={inputs: one_hot_input, expectedQ: targetQ})



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    env = gym.make('FrozenLake-v0')
    # fs_table = FrisbeeSearcher(env=env)
    # successes_history = fs_table.train(**{
    #     'num_episodes':10000, 'gamma':0.85, 'exploration_value':0.1, 'learning_rate':0.85
    # })
    deep_table = DeepFrisbeeSearcher(env=env)
    deep_table.train(num_episodes=100, gamma=0.85)

