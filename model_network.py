import tensorflow as tf

slim = tf.contrib.slim

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
                net = slim.conv2d(inputs=net,
                                  num_outputs=32,
                                  kernel_size=[4, 4],
                                  stride=2,
                                  scope='conv2')
                net = slim.flatten(inputs=net,
                                   scope='flatten1')
                net = slim.fully_connected(inputs=net,
                                           num_outputs=256,
                                           scope='fullyconn1')
                net = slim.fully_connected(inputs=net,
                                           num_outputs=nactions,
                                           activation_fn=None,
                                           scope='fullyconn2')
                net = slim.softmax(net,
                                   scope='final_Q_value')
    return net