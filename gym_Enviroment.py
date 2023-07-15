# gym Enviroment 
from DQN_agent import DQN
from time import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gym
class Gym_Enviroment:
    def __init__(self, 
                 env:gym.Env, 
                 seed:int=None, 
                 rho:float=0.9, 
                 lamda: float=0.99, 
                 epsilon: float=0.9, 
                 epsilon_decay: float=0.999, 
                 epsilon_min:float=0.01, 
                 test:bool=False, 
                 batch_size:int=64, 
                 reply_memory_size:int=100_000, 
                 n_episode:int=10_000, 
                 sample_rate:int=100, 
                 temp_folder:str='./temp/',
                 model_folder:str='./model/'):

        self.dqn = DQN(n_state=env.observation_space.high.shape[0], n_action=env.action_space.n, seed=seed, rho=rho, lamda=lamda, 
                       epsilon=epsilon,epsilon_decay=epsilon_decay,epsilon_min=epsilon_min,
                       test=test,batch_size=batch_size,reply_memory_size=reply_memory_size)

        self.env = env # 학습 환경
        self.seed = seed
        self.n_episode = n_episode # train episode number

        self.defalut_epsilon = epsilon

        self.total_train_time = 0
        self.episode_100_time = 0

        self.sample_rate = sample_rate

        self.scores = []

        self.temp_folder = temp_folder
        self.model_folder = model_folder

    def timedelta_sec2hhmmss(self,old_time): return str(datetime.timedelta(seconds=time()-old_time)).split('.')[0]
    
    def train_episode(self,episode_index):
        state = self.env.reset(seed=self.seed)[0]
        done = False
        episode_time = time()
        while not done:
            action = self.dqn.predict(state) 
            update_state, reward, terminated, truncated, info = self.env.step(action) 
            done = terminated or truncated

            reward = reward + (1 if terminated else 0)
            
            log = 'episode %d | episode train time %s | total train time %s | steps %d | epsilon %.3f | reward %.3f | loss mean %.3f | loss %.3f'\
                %(episode_index,self.timedelta_sec2hhmmss(episode_time),self.timedelta_sec2hhmmss(self.total_train_time),
                  self.env._elapsed_steps,self.dqn.epsilon,reward,np.array(self.dqn.losses_list).mean(),self.dqn.losses_list[-1])
            print(log,end='\r')
            with open(self.model_folder+'log.txt','a') as f: f.write(log+'\n')
            with open(self.temp_folder+'log.txt','a') as f: f.write(log+'\n')
            np.save(self.temp_folder+'env_rgb',self.env.render())
            with open(self.temp_folder+'env_info.txt','w') as f:   
                f.write('%.3f | %.3f | %.3f | Episode: %d | Step: %d'
                        %(update_state[0],update_state[1],reward,episode_index,self.env._elapsed_steps))
 
            self.dqn.remember(state,action,reward,update_state,done)
            self.dqn.model_learning()
            state = update_state

        self.scores.append(self.env._elapsed_steps)

        with open(self.temp_folder+'score_log.txt','a') as f: f.write('%d %d\n'%(episode_index,self.env._elapsed_steps))
        with open(self.model_folder+'score_log.txt','a') as f: f.write('%d %d\n'%(episode_index,self.env._elapsed_steps))
        if episode_index and episode_index%self.sample_rate == 0: 
            self.dqn.save_model(self.model_folder+"DQN_episode_"+'{:04d}'.format(episode_index)+".h5")
            print('current episode:%5d'%episode_index,
                  '| last %d episode mean score: %4d'%(self.sample_rate,np.mean(self.scores[-self.sample_rate:])),
                  '| last %d episode train time:'%self.sample_rate,self.timedelta_sec2hhmmss(self.episode_100_time),
                  '| total train time: ',self.timedelta_sec2hhmmss(self.total_train_time),
                  '| loss mean %.3f'%np.array(self.dqn.losses_list).mean(),
                  '| loss %.3f'%self.dqn.losses_list[-1])
            self.episode_100_time = time()
           
        self.dqn.garbage_collector()

    def train(self):
        self.total_train_time = time()
        self.episode_100_time = time()
        with open(self.temp_folder+'score_log.txt','w') as f: f.write('\n')
        with open(self.model_folder+'score_log.txt','w') as f: f.write('\n')
        with open(self.model_folder+'log.txt','w') as f: f.write('\n')
        for i in range(self.n_episode):
            self.train_episode(i+1)
    
    def test(self):
        state = self.env.reset(seed=self.seed)[0]
        done = False 
        while not done: 
            action = self.dqn.predict(state)
            update_state,reward,terminated,truncated,_ = self.env.step(action)
            done = terminated or truncated
            np.save(self.temp_folder+'env_rgb',self.env.render())
            with open(self.temp_folder+'env_info.txt','w') as f:   
                f.write('%.3f | %.3f | %.3f | Episode: %d | Step: %d'
                        %(update_state[0],update_state[1],reward,self.env._elapsed_steps))
        
    def score_plot(self):
        plt.plot(range(1,len(self.scores)+1),self.scores)
        plt.title('DQN scores')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.grid()
        plt.show()

        plt.plot(range(1,len(self.dqn.losses_list)+1),self.dqn.losses_list)
        plt.title('DQN losses')
        plt.ylabel('Loss')
        plt.xlabel('Episode')
        plt.grid()
        plt.show()