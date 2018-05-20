import os
import gym
import random
import numpy as np
import tensorflow as tf
import threading as thread
import model_atari_game as game
from model_network import deep_Q_network


FLAGS = tf.app.flags.FLAGS


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
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to run asynch training, select 1 for DEBUG')


T = 0
TMAX = 1000000


THREADS = FLAGS.threads


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    final_eps = np.random.choice(final_epsilons, 1, p=list(probabilities))[0]
    return 0.9


def train_mythreads(training_arguments, thread_id):

    # arguments
    sess, runtime_log, reset_target_network_params, \
    game_env, qValue, deepqn, n_actions, target_qValue, \
    target_deepqn, grads, y_target, action_, saver, \
    summary_op, update_ops, summary_placeholders, T, writer = training_arguments

    # epsilon parameters
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

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
            q_value = sess.run([qValue], feed_dict={deepqn: [state_t]})
            # action parameter
            action_t = np.zeros([n_actions])

            e = final_epsilon

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
                print("THREAD ID:",thread_id, "TIME:", T, "/EPSILON", epsilon, "/RWD:", per_episode_reward, "/Q_MAX:%.4f" % (
                    per_episode_average_maxq / float(per_episode_t)), "/E PRGS:", T / 1000000)
                break


def train():

    # compile game environments
    g_env = [gym.make(FLAGS.game) for i in range(THREADS)]

    # declare counter as global
    global T

    # pin the complete process to the CPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # game environment
        game_envs = []
        for tid in range(THREADS):
            game_envs.append(game.Atari(g_env[tid], FLAGS.height, FLAGS.width, FLAGS.collateframes, FLAGS.game))

        # build network
        with tf.name_scope('input'):
            deepqn = tf.placeholder(tf.float32,
                                    [None, FLAGS.width, FLAGS.height, FLAGS.collateframes],
                                    name="deepq-state")
            target_deepqn = tf.placeholder(tf.float32,
                                           [None, FLAGS.width, FLAGS.height, FLAGS.collateframes],
                                           name="target-deepq-state")
        # calculate the number of actions
        n_actions = game_envs[0].n_actions

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

        # instantiate a tensorflow session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # tensorflow log writer
        writer = tf.summary.FileWriter(runtime_log, sess.graph)
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Initialize target network weights
        sess.run(reset_target_network_params)

        # compile all training parameters
        training_arguments = []
        for game_env in game_envs:
            training_arguments.append([sess, runtime_log, reset_target_network_params,
                                       game_env, qValue, deepqn, n_actions, target_qValue,
                                       target_deepqn, grads, y_target, action_, saver,
                                       summary_op, update_ops, summary_placeholders, T, writer])

        # train threads
        train_threads = [thread.Thread(target=train_mythreads, args=(training_arguments[tid], tid, )) for tid in range(THREADS)]
        # start threads
        for t_ in train_threads:
            print("Begin Training with thread:", t_)
            t_.start()

        # show emulator console during training
        while True:
            if FLAGS.show_training:
                for env in g_env:
                    env.render()

        # Join threads to end training
        for t_ in train_thread:
            t_.join()



if __name__ == "__main__":
    # begin training
    train()



