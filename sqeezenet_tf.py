import argparse

import numpy as np
import tensorflow as tf

'''
  parsing and configuration
'''


def parse_args():
    desc = "TensorFlow implementation of 'A Neural Algorithm for Artisitc Style'"
    parser = argparse.ArgumentParser(description=desc)

    # options for single image
    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag indicating if statements should be printed to the console.')
    args = parser.parse_args()

    return args


def neural_network(x):
    net = {}
    # _, h, w, d = input_img
    _, h, w, d = [-1, 224, 224, 3]

    # input - placeholder: add batch size to 1st layer later
    net['input'] = tf.placeholder(tf.float32, [batch_size, h, w, d])

    # conv1_1
    net['conv1'] = conv_layer('conv1', net['input'], W=weight_variable([3, 3, 3, 64], name='conv1'),
                              stride=[1, 2, 2, 1])
    net['relu1'] = relu_layer('relu1', net['conv1'], b=bias_variable([64]))

    net['pool1'] = pool_layer('pool1', net['relu1'])

    # fire2

    net['fire2'] = fire_module('fire2', net['pool1'], 16, 64, 64)
    net['fire3'] = fire_module('fire3', net['fire2'], 16, 64, 64)
    # maxpool
    net['pool3'] = pool_layer('pool3', net['fire3'])

    net['fire4'] = fire_module('fire4', net['pool3'], 32, 128, 128)
    net['fire5'] = fire_module('fire5', net['fire4'], 32, 128, 128)
    net['pool5'] = pool_layer('pool5', net['fire5'])

    net['fire6'] = fire_module('fire6', net['pool5'], 48, 192, 192)
    net['fire7'] = fire_module('fire7', net['fire6'], 48, 192, 192)
    net['fire8'] = fire_module('fire8', net['fire7'], 64, 256, 256)
    net['fire9'] = fire_module('fire9', net['fire8'], 64, 256, 256)
    # 50% dropout
    keep_prob = tf.placeholder(tf.float32)
    net['dropout9'] = tf.nn.dropout(net['fire9'], keep_prob)

    net['conv10'] = conv_layer('conv10', net['dropout9'],
                               W=weight_variable([1, 1, 512, 1000], name='conv10', init='gauss'))
    net['relu10'] = relu_layer('relu10', net['conv10'], b=bias_variable([1000]))

    net['pool10'] = pool_layer('pool10', net['relu10'], pooling_type='avg')

    return net


def conv_layer(layer_name, layer_input, W, stride=[1, 1, 1, 1]):
    conv = tf.nn.conv2d(layer_input, W, strides=stride, padding='SAME')
    if args.verbose: print('--{} | shape={} | weights_shape={} '
                           .format(layer_name, conv.get_shape(), W.get_shape()))
    return conv


def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    if args.verbose:
        print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(),
                                                       b.get_shape()))
    return relu


def pool_layer(layer_name, layer_input, pooling_type='max'):
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 13, 13, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    if args.verbose:
        print('--{}   | shape={} '.format(layer_name, pool.get_shape()))
    return pool


def fire_module(layer_name, layer_input, s1x1, e1x1, e3x3):
    fire = {}

    shape = layer_input.get_shape()

    s1_weight = weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')
    e1_weight = weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
    e3_weight = weight_variable([1, 1, s1x1, e3x3], layer_name + '_e3')

    fire['s1'] = conv_layer(layer_name + '_s1', layer_input, W=s1_weight)
    fire['relu1'] = relu_layer(layer_name + '_relu1', fire['s1'], b=bias_variable([s1x1]))

    fire['e1'] = conv_layer(layer_name + '_e1', fire['relu1'], W=e1_weight)
    fire['e3'] = conv_layer(layer_name + '_e3', fire['relu1'], W=e3_weight)
    fire['concat'] = tf.concat(3, [fire['e1'], fire['e3']])
    if args.verbose:
        print('--{}   | shape={} '.format(layer_name + '_concat', fire['concat'].get_shape()))

    fire['relu2'] = relu_layer(layer_name + '_relu2', fire['concat'], b=bias_variable([e1x1 + e3x3]))

    if args.verbose:
        print('--{}   | shape={} '.format(layer_name, fire['relu2'].get_shape()))

    return fire['relu2']


def weight_variable(shape, name=None, init='xavier'):
    if init == 'xavier':
        initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
    else:
        initial = tf.Variable(tf.truncated_normal(shape, stddev=0.01))

    return initial


def weight_xavier(shape, num_in, num_out):
    low = -4 * np.sqrt(6.0 / (num_in + num_out))  # {sigmoid:4, tanh:1}
    high = 4 * np.sqrt(6.0 / (num_in + num_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


num_examples = 1
batch_size = 512


def train(x):
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.inverse_time_decay(0.04, global_step, )

    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 1000
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(num_examples // batch_size):
                # img, label
                epoch_x, epoch_y = next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', num_epochs, 'losses', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))#reshape

        # accuracy = tf.reduce_mean(tf.cast)

def save_weights():
    print('OMG I FORGOT TO SAVE THE WEIGHTS!')


def main():
    global args
    args = parse_args()

    model = neural_network(None)


if __name__ == '__main__':
    main()
