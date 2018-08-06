import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from network import make_network, make_convs

def build_rp_loss(convs, padding, rp_frame, obs, reward_tp1):
    # encode inputs
    with tf.variable_scope('model'):
        encodes = layers.flatten(make_convs(obs, convs, padding, reuse=True))
    out = tf.reshape(encodes, [1, -1])
    out = layers.fully_connected(out, 128, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 3, activation_fn=None)

    # reward one hot vector
    reward_tp1_one_hot = tf.one_hot(reward_tp1, 3, dtype=tf.float32)
    reward_tp1_one_hot = tf.reshape(reward_tp1_one_hot, [1, -1])

    # compute cross entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=reward_tp1_one_hot, logits=out)
    return loss

def build_vr_loss(convs,
                  fcs,
                  padding,
                  lstm,
                  obs_t,
                  actions_tm1,
                  rewards_t,
                  num_actions,
                  lstm_unit,
                  returns_t):
    init_state = tf.zeros((1, lstm_unit), dtype=tf.float32)
    rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
    _, value_t, _ = make_network(convs, fcs, padding, lstm, obs_t, actions_tm1,
                               rewards_t, rnn_state_tuple, num_actions,
                               lstm_unit, scope='model', reuse=True)
    returns_t = tf.reshape(returns_t, [-1, 1])
    loss = tf.reduce_sum((returns_t - value_t) ** 2)
    return loss

def build_train(convs,
                fcs,
                padding,
                lstm,
                num_actions,
                optimizer,
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                policy_factor=1.0,
                entropy_factor=0.01,
                rp_frame=3,
                scope='a3c',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # placeholers
        obs_t_ph = tf.placeholder(tf.float32, [None] + state_shape, name='obs_t')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_1')
        actions_tm1_ph = tf.placeholder(tf.int32, [None], name="action_tm1")
        rewards_t_ph = tf.placeholder(tf.float32, [None], name="reward_t")

        # placeholders for A3C update
        actions_t_ph = tf.placeholder(tf.uint8, [None], name='action_t')
        returns_t_ph = tf.placeholder(tf.float32, [None], name='return_t')
        advantages_t_ph = tf.placeholder(tf.float32, [None], name='advantage_t')

        # placeholders for reward prediction update
        rp_obs_ph = tf.placeholder(tf.float32, [rp_frame] + state_shape, name='rp_obs')
        rp_reward_tp1_ph = tf.placeholder(tf.int32, [], name='rp_reward_tp1')

        # placeholders for value function replay update
        vr_obs_t_ph = tf.placeholder(tf.float32, [None] + state_shape, name='vr_obs_t')
        vr_actions_tm1_ph = tf.placeholder(tf.int32, [None], name='vr_action_tm1')
        vr_rewards_t_ph = tf.placeholder(tf.float32, [None], name='vr_reward_t')
        vr_returns_t_ph = tf.placeholder(tf.float32, [None], name='vr_returns_t')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        actions_tm1_one_hot = tf.one_hot(
            actions_tm1_ph, num_actions, dtype=tf.float32)
        policy_t, value_t, state_out = make_network(
            convs, fcs, padding, lstm, obs_t_ph, actions_tm1_one_hot,
            rewards_t_ph, rnn_state_tuple, num_actions, lstm_unit, scope='model')

        actions_t_one_hot = tf.one_hot(actions_t_ph, num_actions, dtype=tf.float32)
        log_policy_t = tf.log(tf.clip_by_value(policy_t, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy_t * actions_t_one_hot, axis=1, keep_dims=True)

        # A3C loss
        advantages_t = tf.reshape(advantages_t_ph, [-1, 1])
        returns_t = tf.reshape(returns_t_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_sum((returns_t - value_t) ** 2)
        with tf.variable_scope('entropy_penalty'):
            entropy = -tf.reduce_sum(policy_t * log_policy_t)
        with tf.variable_scope('policy_loss'):
            policy_loss = tf.reduce_sum(log_prob * advantages_t)
        a3c_loss = value_factor * value_loss\
            - policy_factor * policy_loss - entropy_factor * entropy

        # reward prediction loss
        rp_loss = build_rp_loss(
            convs, padding, rp_frame, rp_obs_ph, rp_reward_tp1_ph)

        vr_actions_tm1_one_hot = tf.one_hot(
            vr_actions_tm1_ph, num_actions, dtype=tf.float32)
        vr_loss = build_vr_loss(convs, fcs, padding, lstm, vr_obs_t_ph,
                                vr_actions_tm1_one_hot, vr_rewards_t_ph,
                                num_actions, lstm_unit, vr_returns_t_ph)

        # final loss
        loss = a3c_loss + rp_loss + vr_loss

        # local network weights
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        # global network weights
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        # gradients
        gradients = tf.gradients(loss, local_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)

        optimize_expr = optimizer.apply_gradients(zip(gradients, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)

        def update_local(sess=None):
            if sess is None:
                sess = tf.get_default_session()
            sess.run(update_local_expr)

        def train(obs_t, rnn_state0, rnn_state1, actions_t, rewards_t, actions_tm1,
                  returns_t, advantages_t, rp_obs, rp_reward_tp1, vr_obs_t,
                  vr_actions_tm1, vr_rewards_t, vr_returns_t, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_t_ph: obs_t,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                actions_t_ph: actions_t,
                actions_tm1_ph: actions_tm1,
                rewards_t_ph: rewards_t,
                returns_t_ph: returns_t,
                advantages_t_ph: advantages_t,
                rp_obs_ph: rp_obs,
                rp_reward_tp1_ph: rp_reward_tp1,
                vr_obs_t_ph: vr_obs_t,
                vr_actions_tm1_ph: vr_actions_tm1,
                vr_rewards_t_ph: vr_rewards_t,
                vr_returns_t_ph: vr_returns_t
            }
            loss_val, _ = sess.run([loss, optimize_expr], feed_dict=feed_dict)
            return loss_val

        def act(obs_t, actions_tm1, rewards_t, rnn_state0, rnn_state1, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_t_ph: obs_t,
                actions_tm1_ph: actions_tm1,
                rewards_t_ph: rewards_t,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1
            }
            return sess.run([policy_t, value_t, state_out], feed_dict=feed_dict)

    return act, train, update_local
