import gym
import random
import matplotlib.pyplot as plt
import atari_game as game
import numpy as np
import pandas as pd
import threading as t
import time
import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


slim = tf.contrib.slim


tf.app.flags.DEFINE_string('experiment', 'dqn_breakout', 'Name of the current experiment')
tf.app.flags.DEFINE_string('game', 'Breakout-v0', 'Name of the atari game to play. '
                                                  'Full list here: https://gym.openai.com/envs#atari')
tf.app.flags.DEFINE_integer('height', 84, 'Height of the input frame')
tf.app.flags.DEFINE_integer('width', 84, 'Width of the input frame')
tf.app.flags.DEFINE_integer('collateframes', 4, 'Number of frames to train')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Number of frames to train')
tf.app.flags.DEFINE_float('gamma', 0.99, 'Future reward')
tf.app.flags.DEFINE_string('logdir', os.getcwd()+"/logdir", 'Name of the current experiment')
tf.app.flags.DEFINE_string('checkpoint', os.path.join(FLAGS.logdir, FLAGS.experiment, "checkpoint"),
                           'Height of the frame to retain and train')
tf.app.flags.DEFINE_integer('network_update_n', 32, 'Update network every n steps')
tf.app.flags.DEFINE_integer('target_network_update_frequency', 10000, 'Update target network every n steps')
tf.app.flags.DEFINE_bool('show_training', True, 'Display training')

T = 0
TMAX = 1000000

THREADS = 1


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def deep_Q_network(input_state, nactions, name_scope):
    with tf.device("/cpu:0"):
        with tf.variable_scope(name_scope, 'dqn', [input_state]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.flatten, slim.fully_connected],
                                outputs_collections=end_points_collection):

                net = slim.conv2d(inputs=input_state,
                                  num_outputs=16,
                                  kernel_size=[8, 8],
                                  stride=4,
                                  scope='conv1')
                print(net.get_shape())
                net = slim.conv2d(inputs=net,
                                  num_outputs=32,
                                  kernel_size=[4, 4],
                                  stride=2,
                                  scope='conv2')
                print(net.get_shape())
                net = slim.flatten(inputs=net,
                                   scope='flatten1')
                print(net.get_shape())
                net = slim.fully_connected(inputs=net,
                                           num_outputs=256,
                                           scope='fullyconn1')
                print(net.get_shape())
                net = slim.fully_connected(inputs=net,
                                           num_outputs=nactions,
                                           activation_fn=None,
                                           scope='Qvalue')
                print(net.get_shape())
    return net


def train(g_env):
    global T
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # game environment
        game_env = game.Atari(g_env, FLAGS.height, FLAGS.width, FLAGS.collateframes, FLAGS.game)

        # build network
        with tf.name_scope('input'):
            deepqn = tf.placeholder(tf.float32,
                                    [None, FLAGS.width, FLAGS.height, FLAGS.collateframes],
                                    name="deepq-state")
            target_deepqn = tf.placeholder(tf.float32,
                                           [None, FLAGS.width, FLAGS.height, FLAGS.collateframes],
                                           name="target-deepq-state")
        # calculate the number of actions
        n_actions = game_env.n_actions

        # trainable network
        qValue = deep_Q_network(deepqn, n_actions, "deep-q-network")
        # grab network parameters
        network_params = tf.trainable_variables(scope="deep-q-network")
        # target network
        target_qValue = deep_Q_network(target_deepqn, n_actions, "target-deep-q-network")
        # grab target network parameters
        target_network_params = tf.trainable_variables(scope="target-deep-q-network")
        # assign shared network value to target
        reset_target_network_params = [target_network_params[i].assign(network_params[i])
                                       for i in range(len(target_network_params))]

        # dummy controller action
        action_ = tf.placeholder(tf.float32, [None, n_actions])
        # target reward
        y_target = tf.placeholder(tf.float32, [None])
        # action cost
        q_value_action_cost = tf.reduce_mean(tf.multiply(qValue, action_), reduction_indices=1)
        # loss/cost
        loss_cost = tf.reduce_mean(tf.squared_difference(y_target, q_value_action_cost))
        # optimizer
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # gradient descent
        grads = optimizer.minimize(loss_cost, var_list=network_params)
        # create a saver
        saver = tf.train.Saver()
        # write log
        runtime_log = os.path.join(FLAGS.logdir, FLAGS.experiment)
        # make checkpoint dirs
        os.makedirs(FLAGS.checkpoint, exist_ok=True)
        # episode reward
        episode_reward = tf.Variable(0.)
        # add to scalar summary
        tf.summary.scalar("Episode_Reward", episode_reward)
        # max q value
        episode_ave_max_q_ = tf.Variable(0.)
        # add to summary
        tf.summary.scalar("Max_Q_Value", episode_ave_max_q_)
        # epsilon value
        logged_epsilon = tf.Variable(0.)
        # add to summary
        tf.summary.scalar("Epsilon", logged_epsilon)

        summary_vars = [episode_reward, episode_ave_max_q_, logged_epsilon]
        summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
        update_ops =[]
        for i in range(len(summary_vars)):
            update_ops.append(summary_vars[i].assign(summary_placeholders[i]))
        # collect all summaries
        summary_op = tf.summary.merge_all()

        # epsilon parameters
        final_epsilon = sample_final_epsilon()
        initial_epsilon = 1.0
        epsilon = 1.0



        # instantiate a tensorflow session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            writer = tf.summary.FileWriter(runtime_log, sess.graph)
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # Initialize target network weights
            sess.run(reset_target_network_params)

            # Initialize network gradients
            state_batch = []
            action_batch = []
            reward_batch = []

            while T < TMAX:

                state_t = game_env.get_initial_frames()
                terminate = False

                per_episode_reward = 0
                per_episode_average_maxq = 0
                per_episode_t = 0

                while True:
                    # use network to obtain q value
                    q_value = sess.run([qValue],feed_dict={deepqn: [state_t]})
                    # action parameter
                    action_t = np.zeros([n_actions])

                    # choose action based on e-greedy policy
                    action_index = 0
                    if random.random() <= epsilon:
                        action_index = random.randrange(n_actions)
                    else:
                        action_index = np.argmax(q_value)
                    action_t[action_index] = 1

                    # Scale down epsilon
                    if epsilon > final_epsilon:
                        epsilon -= (initial_epsilon - final_epsilon) / 1000000

                    state_t_target, reward_t, terminate, _ = game_env.step_in_game(action_index)

                    target_q_value = sess.run([target_qValue], feed_dict={target_deepqn: [state_t_target]})

                    clipped_reward_t = np.clip(reward_t, -1, 1)

                    if terminate:
                        reward_batch.append(clipped_reward_t)
                    else:
                        reward_batch.append(clipped_reward_t + FLAGS.gamma * np.max(target_q_value))
                    action_batch.append(action_t)
                    state_batch.append(state_t)
                    state_t = state_t_target
                    T += 1

                    per_episode_t += 1
                    per_episode_reward += reward_t
                    per_episode_average_maxq += np.max(q_value)

                    if T % FLAGS.network_update_n == 0 or terminate:
                        sess.run(grads, feed_dict={y_target: reward_batch,
                                                   action_: action_batch,
                                                   deepqn: [state_t]})
                        # Initialize network gradients
                        state_batch = []
                        action_batch = []
                        reward_batch = []

                    # Optionally update target network
                    if T % FLAGS.target_network_update_frequency == 0:
                        sess.run(reset_target_network_params)

                    # Save model progress
                    if T % 1000 == 0:
                        saver.save(sess, FLAGS.checkpoint + "/" + FLAGS.experiment + ".ckpt", global_step=T)

                    if T % 100 == 0:
                        summary_str = sess.run(summary_op)
                        writer.add_summary(summary_str, float(T))
                    # Print end of episode stats
                    if terminate:
                        stats = [per_episode_reward, per_episode_average_maxq / float(per_episode_t), epsilon]
                        for i in range(len(stats)):
                            sess.run(update_ops[i], feed_dict={summary_placeholders[i]: float(stats[i])})
                        print("TIME/TIMESTEP", T, "/ EPSILON", epsilon, "/ REWARD", per_episode_reward, "/ Q_MAX %.4f" % (
                            per_episode_average_maxq / float(per_episode_t)), "/ EPSILON PROGRESS", T / 1000000)
                        break
                    





if __name__ == "__main__":
    envs = [gym.make(FLAGS.game) for i in range(THREADS)]
    train_thread = [t.Thread(target=train, args=(envs[0],)) for thread_id in range(THREADS)]
    for t in train_thread:
        print("Begin Training")
        t.start()

    # envs = [gym.make(FLAGS.game) for i in range(THREADS)]

    while True:
        if FLAGS.show_training:
            for env in envs:
                env.render()

    for t in train_thread:
        t.join()




