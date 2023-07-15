# dqn agent
import numpy as np 
import random 
import gc
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):self.losses = []
    def on_batch_end(self, batch, logs={}):self.losses.append(logs.get('loss'))

class DQN:
    def __init__(self, 
                 n_state:int, 
                 n_action:int, 
                 seed:int=None, 
                 rho: float=0.9, 
                 lamda:float=0.99, 
                 epsilon:float=0.9, 
                 epsilon_decay:float=0.999, 
                 epsilon_min:float=0.01, 
                 test:bool=False, 
                 batch_size:int=64, 
                 reply_memory_size:int=100_000):

        self.set_random_seed(seed)

        self.rho = rho # 학습률
        self.lamda = lamda # 할인율
        self.epsilon = 0 if test else epsilon # epsilon
        self.default_epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay # epsilon 감소율
        self.epsilon_min = epsilon_min # epsilon minimum
        
        self.batch_size = batch_size # mini batch size

        self.model = self.build_model(input_dim=n_state,output_dim=n_action)
        self.n_state = n_state
        self.n_action = n_action

        self.memory = deque(maxlen=reply_memory_size)

        self.history = LossHistory()
        self.losses_list=[0]

    def set_random_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_model(self, input_dim, output_dim, units=256):
        model = Sequential()
        model.add(Dense(units, input_dim=input_dim, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(units, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(output_dim, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        return model

    def model_learning(self):
        if len(self.memory) > (self.batch_size*3):
            mini_batch = np.array(random.sample(self.memory,self.batch_size),dtype=object)
            state = np.array([mini_batch[i,0] for i in range(self.batch_size)])
            update_state = np.array([mini_batch[i,3] for i in range(self.batch_size)])

            target = self.model.predict(state,verbose=0)
            update_target = self.model.predict(update_state,verbose=0)
        
            for i,(s,a,r,us,d) in enumerate(mini_batch): 
                # s: state, a: action, r: reward, us: update_state, d: done
                if d: target[i][a] = r
                else: target[i][a] += self.rho*((r+self.lamda*np.amax(update_target[i]))-target[i][a])
            self.model.fit(state,target,batch_size=self.batch_size,epochs=1,verbose=0,callbacks=[self.history])
            self.losses_list.append(self.history.losses[0])
    
    def epsilon_decay(self): return max(self.epsilon_min,self.epsilon*self.epsilon_decay_rate) 

    def q(self,state): return self.model.predict(np.reshape(state,(1,self.n_state)),verbose=0)
    
    def predict(self,state):
        if self.epsilon != 0: self.epsilon = self.epsilon_decay()
        if (np.random.random() < self.epsilon): return np.random.randint(0,self.n_action)
        else: return np.argmax(self.q(state)[0])
    
    def remember(self,state,action,reward,update_state,done): self.memory.append((state,action,reward,update_state,done))

    def garbage_collector(self):# tensorflow memory leak 해결(predict) 
        tf.keras.backend.clear_session()
        gc.collect()

    def save_model(self, file_name): self.model.save(file_name) 
    def load_model(self, file_name): self.model = tf.keras.models.load_model(file_name)