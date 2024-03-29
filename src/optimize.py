import functools
import vgg, pdb, time, os
import tensorflow as tf
import numpy as np
import transform
from utils import get_img

STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
CONTENT_LAYER = 'conv4_2'

def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000, batch_size=4,
             checkpoint_dir='saver/fns.ckpt', learning_rate=1e-3):

    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod]  # discard the remaining

    batch_shape = (batch_size, 256, 256, 3)

    style_features = _style_features(style_target, vgg_path)

    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    X_pre = vgg.preprocess(X_content)

    content_features = {}
    content_net = vgg.net(vgg_path, X_pre)
    content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

    preds = transform.net(X_content / 255.0)
    preds_pre = vgg.preprocess(preds)
    net = vgg.net(vgg_path, preds_pre)

    # compute loss
    content_loss = _content_loss(content_weight, net, content_features, batch_size)
    style_loss = _style_loss(style_weight, net, style_features)
    tv_loss = _tv_loss(tv_weight, preds, batch_shape)
    loss = content_loss + style_loss + tv_loss
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                    X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)  # resize content image

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                    X_content: X_batch
                }

                train_step.run(feed_dict = feed_dict)

                is_print_iter = int(iterations) % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {X_content: X_batch}
                    tup = sess.run(to_get, feed_dict=test_feed_dict)
                    style_loss_p, content_loss_p, tv_loss_p, loss_p, preds_p = tup
                    losses = (style_loss_p, content_loss_p, tv_loss_p, loss_p)
                    yield (preds_p, losses, iterations, epoch)


def _style_features(style_target, vgg_path):

    style_features = {}
    style_shape = (1,) + style_target.shape
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_placeholder = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_placeholder_pre = vgg.preprocess(style_placeholder)
        # pdb.set_trace()
        net = vgg.net(vgg_path, style_placeholder_pre)
        style_target_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_placeholder: style_target_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    return style_features


def _content_loss(content_weight, net, content_features, batch_size):

    content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size
    assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
    return content_weight * (2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)


def _style_loss(style_weight, net, style_features):

    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        # bs, height, width, filters = map(lambda i:i.value, layer.get_shape())
        bs, height, width, filters = [i.value for i in layer.shape]
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0, 2, 1])
        # predictions' style feature
        # bs * channels * channels
        grams = tf.matmul(feats_T, feats) / size
        style_gram = style_features[style_layer]  # style features
        style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
    style_loss = style_weight * functools.reduce(tf.add, style_losses) / bs
    return style_loss


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def _tv_loss(tv_weight, preds, batch_shape):
    tv_y_size = _tensor_size(preds[:, 1:, :, :])
    tv_x_size = _tensor_size(preds[:, :, 1:, :])
    y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
    x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
    return tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_shape[0]
