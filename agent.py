from __future__ import print_function
import sys
import time
import random
import numpy as np
import pandas as pd
from math import floor
from collections import deque
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
import keras
import tensorflow as tf
from constants import *
from CONV_GRU import modelBuilder
from tqdm import tqdm

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

class Agent(object):
    def __init__(self, processor):
        self.action_list = ACTIONS
        self.processor = processor
        # first-in-first-out, keep a fixed size of recent history
        # element structure: state_index, action, reward, done
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.model = modelBuilder.build((NUM_CHANNELS, HISTORY_LENGTH + 1), 1)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss = huber_loss, optimizer = sgd) # learning rate default to 0.01
        #self.model.load_weights('logs/model.h5')
        self.logs_df = pd.DataFrame(columns = ['loss', 'lr', 'epsilon'])
        self.test_df = pd.DataFrame(columns = ['action', 'true', 'price'])
    
    """
    Return all the channels from state_idex-HISTORY_LENGTH+1 to state_index.
    """
    def get_state(self, state_index):
        state = [self.processor.get_channels(i) for i in range(state_index - HISTORY_LENGTH + 1, state_index + 1)]
        return state

    """
    Predict on the single data at state_index.
    """
    def predict(self, state_index, action):
        state = np.expand_dims(self.concate_state_action(state_index, action), axis=0)
        prediction = self.model.predict(state)[0]
        return prediction

    """
    Reward function.
    """
    
    def reward_func(self, state_index, action):
        if action == 0: # buy
            reward = (self.processor.price[state_index+1] / self.processor.price[state_index] - 1)
        else:
            reward = -(self.processor.price[state_index+1] / self.processor.price[state_index] - 1)
        return reward
    
    """
    Act base on the currenty state.
    """
    def act(self, state_index):
        # random.random() return a float between 0.0 and 1.0
        if random.random() < self.epsilon: # explore
            action = random.randrange(NUM_ACTIONS)
        else: # exploit
            q_values = np.array([self.predict(state_index, 0), self.predict(state_index, 1)])
            action = q_values.argmax()
        reward = self.reward_func(state_index, action)
        return action, reward
    
    def concate_state_action(self, state_index, action):
        a = np.zeros(NUM_CHANNELS, dtype = int)
        a += action
        a = np.array([a])
        #print(np.shape(a))
        state = self.get_state(state_index)
        #print(np.shape(state))
        return np.concatenate((state, a), axis = 0)
    
    """
    Randomly sample a batch from the replay memory
    """
    def replay_batch(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)
        X_batch, Y_batch = [], []
        for state_index, action, reward in minibatch:
            X_batch.append(self.concate_state_action(state_index, action))
            q_values = np.array([self.predict(state_index+1, 0), self.predict(state_index+1, 1)])
            target = reward + DISCOUNT_FACTOR * np.amax(q_values)
            Y_batch.append(target)
        Y_batch = np.array(Y_batch)
        X_batch = np.array(X_batch)
        return X_batch, Y_batch
    
    def train(self):
        num_data = self.processor.num_data
        num_train_data = floor((1-TEST_FRAC)*num_data)
        steps_per_epoch = floor(num_train_data/BATCH_SIZE)
        print('Use {} data for training.'.format(num_train_data))
        print('There are {} steps per epoch.'.format(steps_per_epoch))
        # build memory/experience: optimal policy for train data
        for state_index in range(HISTORY_LENGTH, num_train_data):
            if self.processor.price[state_index] <= self.processor.price[state_index+1]:
                action = 0
            else:
                action = 1
            reward = self.reward_func(state_index, action)
            self.memory.append((state_index, action, reward))
        # learn from memory
        for epoch in range(NUM_EPOCHS):
            history = []
            for step in tqdm(range(steps_per_epoch)):  
                if len(self.memory) > BATCH_SIZE:
                    X_batch, Y_batch = self.replay_batch()
                    loss = self.model.train_on_batch(X_batch, Y_batch)
                    lr = K.get_value(self.model.optimizer.lr)
                    # record loss and learning rate
                    history.append([loss, lr])
                    # do epsilon decay per EPSILON_UPDATE_STEP
                    if self.epsilon > self.epsilon_min:
                        if (epoch*steps_per_epoch + step) % EPSILON_UPDATE_STEP == EPSILON_UPDATE_STEP - 1:
                            self.epsilon *= EPSILON_DECAY
                    else:
                        self.epsilon = self.epsilon_min
            self.test()
            # do learning rate decay
            #if epoch % LR_UPDATE_EPOCH == LR_UPDATE_EPOCH - 1:
            #    K.set_value(self.model.optimizer.lr, lr*LR_DECAY)
            history = np.mean(history, axis = 0)
            print('epoch {}: loss: {}, learning rate: {}, epsilon: {}'.format(epoch, history[0], history[1], self.epsilon))
            df = pd.DataFrame([np.append(history, self.epsilon)], columns = ['loss', 'lr', 'epsilon'])
            self.logs_df = self.logs_df.append(df, ignore_index=True)
            self.logs_df.to_csv('logs/history.csv')
            self.model.save_weights('logs/model.h5')
    
    def test(self):
        epsilon = self.epsilon
        self.epsilon = 0.0
        num_data = self.processor.num_data
        test_begin = floor((1-TEST_FRAC)*num_data)
        print('Use {} data for testing.'.format(num_data - test_begin))
        actions = []
        for state_index in range(test_begin, num_data-1):
            action, reward = self.act(state_index)
            actions.append(action)

        prices = []
        Y = []
        for state_index in range(test_begin, num_data-1):
            price = self.processor.price[state_index]
            prices.append(price)
            next_price = self.processor.price[state_index+1]
            if next_price > price:
                Y.append(0) # buy
            else:
                Y.append(1) # sell

        c = 0
        for i in range(num_data - test_begin - 1):
            if actions[i] == Y[i]:
                c += 1
        accuracy = c/(num_data - test_begin - 1)
        with open("test/accuracy.txt", 'a') as f:
            f.write('accuracy: {}\n'.format(accuracy))
        
        print('accuracy: {}'.format(accuracy))
        df = pd.DataFrame({'action': actions, 'true': Y, 'price': prices})
        self.test_df = df
        self.test_df.to_csv('test/test.csv')
        self.epsilon = epsilon

