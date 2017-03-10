import tensorflow as tf
import numpy as np
import cv2


class SqueezeNet(object):
    def __init__(self, session, alpha, optimizer=tf.train.GradientDescentOptimizer, squeeze_ratio=1):
        if session:
            self.session = session
        else:
            self.session = tf.Session()

        self.dropout   = tf.placeholder(tf.float32)
        self.target    = tf.placeholder(tf.float32, [None, 1000])
        self.imgs      = tf.placeholder(tf.float32, [None, 224, 224, 3])

        self.alpha = alpha
        self.sq_ratio  = squeeze_ratio
        self.optimizer = optimizer

        self.weights = {}
        self.net = {}

        self.build_model()
        self.init_opt()
        self.init_model()

    def build_model(self):
        net = {}

        # Caffe order is BGR, this model is RGB.
        # The mean values are from caffe protofile from DeepScale/SqueezeNet github repo.
        self.mean = tf.constant([123.0, 117.0, 104.0],
                                dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = self.imgs-self.mean

        net['input'] = images

        # conv1_1
        net['conv1'] = self.conv_layer('conv1', net['input'],
                              W=self.weight_variable([3, 3, 3, 64], name='conv1_w'), stride=[1, 2, 2, 1])

        net['relu1'] = self.relu_layer('relu1', net['conv1'], b=self.bias_variable([64], 'relu1_b'))
        net['pool1'] = self.pool_layer('pool1', net['relu1'])

        net['fire2'] = self.fire_module('fire2', net['pool1'], self.sq_ratio * 16, 64, 64)
        net['fire3'] = self.fire_module('fire3', net['fire2'], self.sq_ratio * 16, 64, 64,   True)
        net['pool3'] = self.pool_layer('pool3', net['fire3'])

        net['fire4'] = self.fire_module('fire4', net['pool3'], self.sq_ratio * 32, 128, 128)
        net['fire5'] = self.fire_module('fire5', net['fire4'], self.sq_ratio * 32, 128, 128, True)
        net['pool5'] = self.pool_layer('pool5', net['fire5'])

        net['fire6'] = self.fire_module('fire6', net['pool5'], self.sq_ratio * 48, 192, 192)
        net['fire7'] = self.fire_module('fire7', net['fire6'], self.sq_ratio * 48, 192, 192, True)
        net['fire8'] = self.fire_module('fire8', net['fire7'], self.sq_ratio * 64, 256, 256)
        net['fire9'] = self.fire_module('fire9', net['fire8'], self.sq_ratio * 64, 256, 256, True)

        # 50% dropout
        net['dropout9'] = tf.nn.dropout(net['fire9'], self.dropout)
        net['conv10']   = self.conv_layer('conv10', net['dropout9'],
                               W=self.weight_variable([1, 1, 512, 1000], name='conv10', init='normal'))
        net['relu10'] = self.relu_layer('relu10', net['conv10'], b=self.bias_variable([1000], 'relu10_b'))
        net['pool10'] = self.pool_layer('pool10', net['relu10'], pooling_type='avg')

        avg_pool_shape        = tf.shape(net['pool10'])
        net['pool_reshaped']  = tf.reshape(net['pool10'], [avg_pool_shape[0],-1])
        self.fc2              = net['pool_reshaped']
        self.logits           = net['pool_reshaped']

        self.probs = tf.nn.softmax(self.logits)
        self.net   = net

    def init_opt(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
        self.optimize = self.optimizer(self.alpha).minimize(self.cost)

    def init_model(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def bias_variable(self, shape, name, value=0.1):
        initial = tf.constant(value, shape=shape)
        self.weights[name] = tf.Variable(initial)
        return self.weights[name]

    def weight_variable(self, shape, name=None, init='xavier'):
        if init == 'variance':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
        elif init == 'xavier':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            initial = tf.Variable(tf.random_normal(shape, stddev=0.01), name='W'+name)

        self.weights[name] = initial
        return self.weights[name]

    def relu_layer(self, layer_name, layer_input, b=None):
        if b:
            layer_input += b
        relu = tf.nn.relu(layer_input)
        return relu

    def pool_layer(self, layer_name, layer_input, pooling_type='max'):
        if pooling_type == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 13, 13, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
        elif pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
        return pool

    def conv_layer(self, layer_name, layer_input, W, stride=[1, 1, 1, 1]):
        return tf.nn.conv2d(layer_input, W, strides=stride, padding='SAME')

    def fire_module(self, layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
        """ Fire module consists of squeeze and expand convolutional layers. """
        fire = {}

        shape = layer_input.get_shape()

        # squeeze
        s1_weight = self.weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')

        # expand
        e1_weight = self.weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
        e3_weight = self.weight_variable([3, 3, s1x1, e3x3], layer_name + '_e3')

        fire['s1'] = self.conv_layer(layer_name + '_s1', layer_input, W=s1_weight)
        fire['relu1'] = self.relu_layer(layer_name + '_relu1', fire['s1'],
                                        b=self.bias_variable([s1x1], layer_name + '_fire_bias_s1'))

        fire['e1'] = self.conv_layer(layer_name + '_e1', fire['relu1'], W=e1_weight)
        fire['e3'] = self.conv_layer(layer_name + '_e3', fire['relu1'], W=e3_weight)
        fire['concat'] = tf.concat([tf.add(fire['e1'], self.bias_variable([e1x1],
                                                           name=layer_name + '_fire_bias_e1' )),
                                    tf.add(fire['e3'], self.bias_variable([e3x3],
                                                           name=layer_name + '_fire_bias_e3' ))], 3)

        if residual:
            fire['relu2'] = self.relu_layer(layer_name + 'relu2_res', tf.add(fire['concat'],layer_input))
        else:
            fire['relu2'] = self.relu_layer(layer_name + '_relu2', fire['concat'])

        return fire['relu2']

    def save_model(self, path):
        """
        Save the neural network model.
        :param path: path where will be stored
        :return: path if success
        """
        saver = tf.train.Saver(self.weights)
        save_path = saver.save(self.session, path)
        return save_path

    def load_model(self, path):
        """
        Load neural network model from path.
        :param path: path where is network located.
        :return: None
        """
        saver = tf.train.Saver(self.weights)
        saver.restore(self.session, path)

if __name__ == '__main__':
    sess = tf.Session()
    alpha= tf.placeholder(tf.float32)
    net  = SqueezeNet(sess, alpha)

    img1 = cv2.imread('./images/architecture.png')#, mode='RGB')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (224, 224))
    prob = sess.run(net.probs, feed_dict={net.net['input']: [img1], net.dropout:1.0})
    print(prob)
    net.save_model('test.ckpt')
    net.load_model('./test.ckpt')
