#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 28
WIDTH = 28
NCLASSES = 10

def linear_model(img, mode, hparams):
    X = tf.reshape(tensor = img, shape = [-1, HEIGHT * WIDTH]) #flatten
    ylogits = tf.layers.dense(inputs = X, units = NCLASSES, activation = None)
    return ylogits, NCLASSES

def dnn_model(img, mode, hparams):
    # TODO: Implement DNN model with three hiddenlayers
    
    # 납작한 모양으로 reshape
    X = tf.reshape(img, [-1, HEIGHT*WIDTH]) # flatten
    
    # TIP: tf.layers를 겹겹이 쌓아주고 이전 레이어를 input으로 주면 됨
    # 히든 레이어: 뉴런의 갯수는 레이어를 거치면서 줄여준다.
    h1 = tf.layers.dense(X, 300, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
    h3 = tf.layers.dense(h2, 30, activation=tf.nn.relu)
    
    # 마지막 layer는 클래수 수를 뉴런의 갯수로 넘겨주고 activation function을 적용하지 않는다
    ylogits = tf.layers.dense(h3, NCLASSES, activation=None)
 
    return ylogits, NCLASSES

def dnn_dropout_model(img, mode, hparams):
    """ TODO: Implement DNN model and apply dropout to the last hidden layer"""
    # DNN 코드를 카피하고 dropout만 하이퍼 파라미터로 추가하면 됨
    # TIP: 10%의 확률로  drop out 하시요
    dprob = hparams.get('dprob', 0.1)
     
    # 납작한 모양으로 reshape
    X = tf.reshape(img, [-1, HEIGHT*WIDTH]) # flatten    
    h1 = tf.layers.dense(X, 300, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
    h3 = tf.layers.dense(h2, 30, activation=tf.nn.relu)
    
    # TIP: 학습하는 과정에만 드랍아웃 해 준다.
    # tf.estimator.ModeKeys.TRAIN
    hrd = tf.layers.dropout(h3, rate=dprob, training=(
        mode == tf.estimator.ModeKeys.TRAIN))
    
    # input change: h3 -> hrd (dropout applied layer)
    ylogits = tf.layers.dense(hrd, NCLASSES, activation=None)
 
    return ylogits, NCLASSES

def cnn_model(img, mode, hparams):
    # 각 레이어에 커널 사이즈 5로 동일하게 넘겨주고 필터수는 다르게 조정해본다.
    ksize1 = hparams.get('ksize1', 5)
    ksize2 = hparams.get('ksize2', 5)

    # 필터: number of distinct kernels
    # 두번 째 레이어는 필터 수를 두 배로 늘려줌
    nfil1 = hparams.get('nfil1', 10)
    nfil2 = hparams.get('nfil2', 20)
    dprob = hparams.get('dprob', 0.25)

    # shape = (batch_size, HEIGHT, WIDTH, nfil1)
    # ?x28x28x10
    c1 = tf.layers.conv2d(inputs = img, filters = nfil1,
                          kernel_size = ksize1, strides = 1, 
                          padding = "same", activation = tf.nn.relu) 
    
    # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil1)
    # ?x14x14x10
    p1 = tf.layers.max_pooling2d(inputs = c1, pool_size = 2, strides = 2) 

    # TODO: apply a second convolution to the output of p1
    # p1을 입력으로 받고 필터 수와 커널수를 레이어2로 바꿔준다. 패딩이나 활성함수는 동일하게 사용했음
    # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil2)
    # ?x14x14x20
    c2 = tf.layers.conv2d(inputs = p1, filters = nfil2,
                          kernel_size = ksize2, strides = 1,
                          padding = "same", activation = tf.nn.relu) 
    # TODO: apply a pooling layer with pool_size = 2 and strides = 2
    # p2 = None # shape = (batch_size, HEIGHT // 4, WIDTH // 4, nfil2)
    # ?x7x7x20
    p2 = tf.layers.max_pooling2d(inputs = c2, pool_size = 2, strides = 2)
  
    # 여기까지 한 후에 사이즈가 맞는지 확인. ?는 배치 사이즈
    #980
    outlen = p2.shape[1] * p2.shape[2] * p2.shape[3] # HEIGHT // 4 * WIDTH // 4 * nfil2
    p2flat = tf.reshape(tensor = p2, shape = [-1, outlen]) # shape = (batch_size, HEIGHT // 4 * WIDTH // 4 * nfil2)

    h3 = tf.layers.dense(inputs = p2flat, units = 300, activation = tf.nn.relu) 
    h3d = tf.layers.dropout(inputs = h3, rate = dprob, training = (mode == tf.estimator.ModeKeys.TRAIN))

    ylogits = tf.layers.dense(inputs = h3d, units = NCLASSES, activation = None)
    return ylogits, NCLASSES

def serving_input_fn():
    # Input will be rank 3
    feature_placeholders = {"image": tf.placeholder(dtype = tf.float32, shape = [None, HEIGHT, WIDTH])}
    # But model function requires rank 4
    features = {"image": tf.expand_dims(input = feature_placeholders["image"], axis = -1)} 
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = feature_placeholders)

def image_classifier(features, labels, mode, params):
    model_functions = {
        "linear": linear_model,
        "dnn": dnn_model,
        "dnn_dropout": dnn_dropout_model,
        "cnn": cnn_model}
    
    model_function = model_functions[params["model"]]
    
    ylogits, nclasses = model_function(features["image"], mode, params)

    probabilities = tf.nn.softmax(logits = ylogits)
    class_ids = tf.cast(x = tf.argmax(input = probabilities, axis = 1), dtype = tf.uint8)
    
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(input_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(logits = ylogits, labels = labels))
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            # This is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss = loss, 
                    global_step = tf.train.get_global_step(),
                    learning_rate = params["learning_rate"], 
                    optimizer = "Adam")
            eval_metric_ops = None
        else:
            train_op = None
            eval_metric_ops =  {"accuracy": tf.metrics.accuracy(labels = tf.argmax(input = labels, axis = 1), predictions = class_ids)}
    else:
        loss = None
        train_op = None
        eval_metric_ops = None
 
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = {"probabilities": probabilities, "class_ids": class_ids},
        loss = loss,
        train_op = train_op,
        eval_metric_ops = eval_metric_ops,
        export_outputs = {"predictions":tf.estimator.export.PredictOutput({"probabilities": probabilities, "class_ids": class_ids})}
    )

def train_and_evaluate(output_dir, hparams):
    tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    
    EVAL_INTERVAL = 60
    
    mnist = input_data.read_data_sets("mnist/data", one_hot = True, reshape = False)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"image": mnist.train.images},
        y = mnist.train.labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True,
        queue_capacity = 5000
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"image": mnist.test.images},
        y = mnist.test.labels,
        batch_size = 100,
        num_epochs = 1,
        shuffle = False,
        queue_capacity = 5000
    )

    estimator = tf.estimator.Estimator(
        model_fn = image_classifier,
        model_dir = output_dir,
        params = hparams)

    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn,
        max_steps = hparams["train_steps"])

    exporter = tf.estimator.LatestExporter(name = "exporter", serving_input_receiver_fn = serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_input_fn,
        steps = None,
        exporters = exporter)

    tf.estimator.train_and_evaluate(estimator = estimator, train_spec = train_spec, eval_spec = eval_spec)