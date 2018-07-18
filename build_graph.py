import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from network import make_network, make_convs

def build_rp_loss(convs, padding, rp_frame, states, reward):
    # encode inputs
    with tf.variable_scope('model'):
        encodes = layers.flatten(make_convs(states, convs, padding, reuse=True))
    out = tf.reshape(encodes, [1, -1])
    out = layers.fully_connected(out, 128, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, 3, activation_fn=None)

    # reward one hot vector
    reward_one_hot = tf.one_hot(reward, 3, dtype=tf.float32)
    reward_one_hot = tf.reshape(reward_one_hot, [1, -1])

    # compute cross entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=reward_one_hot, logits=out)
    return loss

def build_vr_loss(convs,
                  fcs,
                  padding,
                  lstm,
                  obs,
                  last_actions,
                  rewards,
                  num_actions,
                  lstm_unit,
                  target_values):
    init_state = tf.zeros((1, lstm_unit), dtype=tf.float32)
    rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
    _, value, _ = make_network(convs, fcs, padding, lstm, obs, last_actions,
                               rewards, rnn_state_tuple, num_actions,
                               lstm_unit, scope='model', reuse=True)
    target_values = tf.reshape(target_values, [-1, 1])
    loss = tf.reduce_sum((target_values - value) ** 2)
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
        obs_input = tf.placeholder(tf.float32, [None] + state_shape, name='obs')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [1, lstm_unit], name='rnn_state_1')
        last_actions_ph = tf.placeholder(tf.int32, [None], name="last_action")
        rewards_ph = tf.placeholder(tf.float32, [None], name="reward")

        # placeholders for A3C update
        actions_ph = tf.placeholder(tf.uint8, [None], name='action')
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')

        # placeholders for reward prediction update
        rp_obs_ph = tf.placeholder(tf.float32, [rp_frame] + state_shape, name='rp_obs')
        rp_reward_ph = tf.placeholder(tf.int32, [], name='rp_reward')

        # placeholders for value function replay update
        vr_obs_ph = tf.placeholder(tf.float32, [None] + state_shape, name='vr_obs')
        vr_last_actions_ph = tf.placeholder(tf.int32, [None], name='vr_last_action')
        vr_rewards_ph = tf.placeholder(tf.float32, [None], name='vr_reward')
        vr_target_values_ph = tf.placeholder(tf.float32, [None], name='vr_value')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        # network outpus
        last_actions_one_hot = tf.one_hot(
            last_actions_ph, num_actions, dtype=tf.float32)
        policy, value, state_out = make_network(
            convs, fcs, padding, lstm, obs_input, last_actions_one_hot,
            rewards_ph, rnn_state_tuple, num_actions, lstm_unit, scope='model')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, [1])

        # A3C loss
        advantages  = tf.reshape(advantages_ph, [-1, 1])
        target_values = tf.reshape(target_values_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_sum((target_values - value) ** 2)
        with tf.variable_scope('entropy_penalty'):
            entropy = -tf.reduce_sum(policy * log_policy)
        with tf.variable_scope('policy_loss'):
            policy_loss = tf.reduce_sum(log_prob * advantages)
        a3c_loss = value_factor * value_loss\
            - policy_factor * policy_loss - entropy_factor * entropy

        # reward prediction loss
        rp_loss = build_rp_loss(
            convs, padding, rp_frame, rp_obs_ph, rp_reward_ph)

        vr_last_actions_one_hot = tf.one_hot(
            vr_last_actions_ph, num_actions, dtype=tf.float32)
        vr_loss = build_vr_loss(convs, fcs, padding, lstm, vr_obs_ph,
                                vr_last_actions_one_hot, vr_rewards_ph,
                                num_actions, lstm_unit, vr_target_values_ph)

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
        gradients = [tf.clip_by_norm(g, grad_clip) for g in gradients]

        optimize_expr = optimizer.apply_gradients(zip(gradients, global_vars))

        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)

        def update_local(sess=None):
            if sess is None:
                sess = tf.get_default_session()
            sess.run(update_local_expr)

        def train(obs, rnn_state0, rnn_state1, actions, rewards, last_actions,
                  target_values, advantages, rp_obs, rp_reward, vr_obs,
                  vr_last_actions, vr_rewards, vr_target_values, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_input: obs,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1,
                actions_ph: actions,
                last_actions_ph: last_actions,
                rewards_ph: rewards,
                target_values_ph: target_values,
                advantages_ph: advantages,
                rp_obs_ph: rp_obs,
                rp_reward_ph: rp_reward,
                vr_obs_ph: vr_obs,
                vr_last_actions_ph: vr_last_actions,
                vr_rewards_ph: vr_rewards,
                vr_target_values_ph: vr_target_values
            }
            loss_val, _ = sess.run([loss, optimize_expr], feed_dict=feed_dict)
            return loss_val

        def act(obs, action, reward, rnn_state0, rnn_state1, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_input: obs,
                last_actions_ph: action,
                rewards_ph: reward,
                rnn_state_ph0: rnn_state0,
                rnn_state_ph1: rnn_state1
            }
            return sess.run([policy, value, state_out], feed_dict=feed_dict)

    return act, train, update_local
