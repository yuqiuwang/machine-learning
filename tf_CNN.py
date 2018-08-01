# -*- coding: utf-8 -*-
"""
Author  : wangyuqiu
Mail    : yuqiuwang929@gmail.com
Created : 2018/7/30 14:06
"""

import tensorflow as tf
import numpy as np
import glob

#########################################################################################
##读取图片
#########################################################################################


def load_data(my_label="label_1"):

    def read_pic(pic_path):
        image_raw_data = tf.gfile.FastGFile(pic_path, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        img_data = img_data.eval().reshape(300, 300, 3)
        img_data = tf.image.resize_images(img_data, (100, 100))   # 300X300内存需求量过大，调整为100X100
        return img_data.eval()/255.0    # 值控制在[0,1]
    paths = glob.glob("./%s/*.png" % my_label)  # 返回训练图片的路径

    if my_label == "label_1":
        my_label = [0, 1]
    elif my_label == "label_2":
        my_label = [1, 0]

    pictures = []
    pictures_test = []
    labels = []
    labels_test = []
    check_num = 0
    for path in paths:
        check_num += 1
        if check_num % 3 == 0:
            # 三分之二的数据作为训练集，三分之一的数据作为测试集
            pictures_test.append(read_pic(path))
            labels_test.append(my_label)
        else:
            pictures.append(read_pic(path))
            labels.append(my_label)
    return pictures, labels, pictures_test, labels_test

def datas():
    pictures, labels, pictures_test, labels_test = load_data("label_1")
    pictures1, labels1, pictures_test1, labels_test1 = load_data("label_2")
    pictures = np.array(pictures+pictures1)
    pictures_test = np.array(pictures_test+pictures_test1)
    labels = np.array(labels+labels1)
    labels_test = np.array(labels_test+labels_test1)
    #print(np.shape(pictures), np.shape(pictures_test))
    #print(np.shape(labels), np.shape(labels_test))
    return pictures, labels, pictures_test, labels_test


#########################################################################################
##参数
input_shape = [None, 100, 100, 3]
output_node = 2
train_step = 200
#########################################################################################

def weight_variable(shape):
    # 定义权重
    with tf.name_scope('weights'):
        Weights = tf.Variable(tf.random_normal(shape, stddev=0.1))
    return Weights

def biases_variable(shape):
    # 定义偏置
    with tf.name_scope('biases'):
        Biases = tf.Variable(tf.random_normal(shape, mean=0.1, stddev=0.1))
    return Biases

def conv_layer(layername, inputs, Weights_shape, biases_shape, strides=[1, 1, 1, 1], padding='VALID', activation_function=None):  
    # 定义卷积层 
    with tf.name_scope(layername):
        Weights = weight_variable(Weights_shape)
        biases = biases_variable(biases_shape)
        with tf.name_scope("h_conv"):
            h_conv = tf.nn.bias_add(tf.nn.conv2d(inputs, Weights, strides=strides, padding=padding), biases)
        if activation_function is None:
            outputs = h_conv
        else:
            outputs = activation_function(h_conv)
    return outputs

def pool_layer(layername, conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pooling_function=None):
    # 定义池化层
    with tf.name_scope(layername):
        if pooling_function is None:
            outputs = conv
        else:
            outputs = pooling_function(conv, ksize=ksize, strides=strides, padding=padding)
    return outputs

def fc_layer(layername, inputs, Weights_shape, biases_shape, activation_function=None):
    # 定义全连接层
    with tf.name_scope(layername):
        Weights = weight_variable(Weights_shape)
        biases = biases_variable(biases_shape)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, name = layername)
        tf.summary.histogram(layername+"/outputs", outputs)
    return outputs

def mode(inputs, keep_prob):
    # 模型
    conv1_layer1 = conv_layer("conv1_layer1", inputs, [2, 2, 3, 48], [48], [1, 1, 1, 1], 'SAME', tf.nn.relu)
    pool1_layer2 = pool_layer("pooling1_layer2", conv1_layer1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', tf.nn.max_pool)
    conv2_layer3 = conv_layer("conv2_layer3", pool1_layer2, [2, 2, 48, 96], [96], [1, 1, 1, 1], 'SAME', tf.nn.relu)
    pool2_layer4 = pool_layer("pooling2_layer4", conv2_layer3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', tf.nn.max_pool)
    layer4_shape = pool2_layer4.get_shape().as_list()
    print(layer4_shape)
    pool2_layer4flat = tf.reshape(pool2_layer4, [-1, layer4_shape[1]*layer4_shape[2]*layer4_shape[3]])
    fc1_layer5 = fc_layer("fc1_layer5", pool2_layer4flat, [layer4_shape[1]*layer4_shape[2]*layer4_shape[3], 50], [50], tf.nn.relu)
    fc1_layer5_drop = tf.nn.dropout(fc1_layer5, keep_prob)
    fc2_layer6 = fc_layer("fc2_layer6", fc1_layer5_drop, [50, output_node], [output_node])
    return fc2_layer6

def loss(outputs, outputs_target, learning_rate=0.001, Optimizer = "Adam"):
    # 损失函数
    end_points = {}
    # cross_entropy = -tf.reduce_mean(outputs_target*tf.log(outputs))
    cross_entropy = -tf.reduce_mean(outputs_target*tf.log(outputs+1e-10))  # 加上一个极小值，防止log0后出现NAN值
    #outputs = tf.clip_by_value(outputs,1e-10,1)
    #cross_entropy = -tf.reduce_mean(outputs_target*tf.log(outputs))
    if Optimizer == "Adam":
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    elif Optimizer == "GradientDescent":
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(outputs_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    end_points["accuracy"] = accuracy
    end_points["loss"] = cross_entropy
    end_points["train_step"] = train_step
    end_points["outputs"] = outputs
    end_points["outputs_target"] = outputs_target
    return end_points

#######################################################################################
def train():
    ##run tf
    x = tf.placeholder(tf.float32, input_shape, name="data")
    y_ = tf.placeholder(tf.float32, [None, output_node], name="target")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    outputs = tf.nn.softmax(mode(x, keep_prob), name="op_to_store")
    end_points = loss(outputs, y_, 0.005, "GradientDescent")
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        pictures, labels, pictures_test, labels_test = datas()
        sess.run(init_op)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./cnv_tbgragh", sess.graph)
        for i in range(0, train_step):
            _, loss_ = sess.run([end_points["train_step"], end_points["loss"]], feed_dict={x: pictures, y_: labels, keep_prob: 0.5})
            print (i, loss_)
        saver.save(sess, "./cnv_model/cnv_model.ckpt")
        loss_, accuracy_, test_y, true_y = sess.run([\
                        end_points["loss"], end_points["accuracy"], end_points["outputs"], end_points["outputs_target"]], \
                        feed_dict={x: pictures_test, y_: labels_test, keep_prob: 1.0})
        print(accuracy_)
        print(test_y.tolist(), true_y.tolist())

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()

