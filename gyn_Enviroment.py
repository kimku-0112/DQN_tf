# gym Enviroment 
from DQN_agent import DQN

class Gym_Enviroment:
    def __init__(self, env, random_seed=None, rho=0.9, lamda=0.99, epsilon=0.9, epsilon_decay_rate=0.999, epsilon_min=0.01, batch_size=64, 
                 reply_memory_size=100_000, n_episode=10_000, sample_rate=100, temp_folder='./temp/',model_folder='./model/'):
        self.env = env # 학습 환경

        self.random_seed = random_seed
        self.set_random_seed()

        self.rho = rho # 학습률
        self.lamda = lamda # 할인율
        self.epsilon = epsilon # epsilon
        self.epsilon_decay_rate = epsilon_decay_rate # epsilon 감소율
        self.epsilon_min = epsilon_min # epsilon minimum
        
        self.batch_size = batch_size # mini batch size
        self.n_episode = n_episode # train episode number

        self.model = self.make_model()
        self.memory = deque(maxlen=reply_memory_size)

        self.scores = []

        self.total_train_time = 0
        self.episode_100_time = 0

        self.sample_rate = sample_rate

        self.history = LossHistory()
        self.losses_list=[0]

        self.temp_folder = temp_folder
        self.model_folder = model_folder

    def set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def make_model(self, units=256):
        model = Sequential()
        model.add(Dense(units, input_dim=self.env.observation_space.high.shape[0], activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(units, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(self.env.action_space.n, activation='linear', kernel_initializer='he_uniform'))
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

    def q(self,state): return self.model.predict(np.reshape(state,(1,self.env.observation_space.high.shape[0])),verbose=0)

    def choice_action(self,state):return np.argmax(self.q(state)[0])
    
    def predict_action(self,state):
        self.epsilon = self.epsilon_decay()
        if (np.random.random() < self.epsilon): return np.random.randint(0,self.env.action_space.n)
        else: return self.choice_action(state)
    
    def remember(self,state,action,reward,update_state,done): self.memory.append((state,action,reward,update_state,done))

    def reward(self, reward, terminated): return reward + (1 if terminated else 0)

    def timedelta_sec2hhmmss(self,old_time): return str(datetime.timedelta(seconds=time()-old_time)).split('.')[0]
    
    def garbage_collection(self):# tensorflow memory leak 해결(predict) 
        tf.keras.backend.clear_session()
        gc.collect()

    def episode(self,episode_index):
        state = self.env.reset(seed=self.random_seed)[0]
        done = False
        episode_time = time()
        while not done:
            action = self.predict_action(state) 
            update_state, reward, terminated, truncated, info = self.env.step(action) 
            done = terminated or truncated

            reward = self.reward(reward,terminated)
            
            log = 'episode %d | episode train time %s | total train time %s | steps %d | epsilon %.3f | reward %.3f | loss mean %.3f | loss %.3f'\
                %(episode_index,self.timedelta_sec2hhmmss(episode_time),self.timedelta_sec2hhmmss(self.total_train_time),
                  self.env._elapsed_steps,self.epsilon,reward,np.array(self.losses_list).mean(),self.losses_list[-1])
            print(log,end='\r')
            with open(self.model_folder+'log.txt','a') as f: f.write(log+'\n')
            with open(self.temp_folder+'log.txt','a') as f: f.write(log+'\n')
            np.save(self.temp_folder+'env_rgb',self.env.render())
            with open(self.temp_folder+'env_info.txt','w') as f:   
                f.write('%.3f | %.3f | %.3f | Episode: %d | Step: %d'
                        %(update_state[0],update_state[1],reward,episode_index,self.env._elapsed_steps))
 
            self.remember(state,action,reward,update_state,done)
            self.model_learning()
            state = update_state

        self.scores.append(self.env._elapsed_steps)

        with open(self.temp_folder+'score_log.txt','a') as f: f.write('%d %d\n'%(episode_index,self.env._elapsed_steps))
        with open(self.model_folder+'score_log.txt','a') as f: f.write('%d %d\n'%(episode_index,self.env._elapsed_steps))
        if episode_index and episode_index%self.sample_rate == 0: 
            self.save_model(self.model_folder+"DQN_episode_"+'{:04d}'.format(episode_index)+".h5")
            print('current episode:%5d'%episode_index,
                  '| last %d episode mean score: %4d'%(self.sample_rate,np.mean(self.scores[-self.sample_rate:])),
                  '| last %d episode train time:'%self.sample_rate,self.timedelta_sec2hhmmss(self.episode_100_time),
                  '| total train time: ',self.timedelta_sec2hhmmss(self.total_train_time),
                  '| loss mean %.3f'%np.array(self.losses_list).mean(),
                  '| loss %.3f'%self.losses_list[-1])
            self.episode_100_time = time()
           
        self.garbage_collection()

    def train(self):
        self.total_train_time = time()
        self.episode_100_time = time()
        with open(self.temp_folder+'score_log.txt','w') as f: f.write('\n')
        with open(self.model_folder+'score_log.txt','w') as f: f.write('\n')
        with open(self.model_folder+'log.txt','w') as f: f.write('\n')
        for i in range(self.n_episode):
            self.episode(i+1)
            # if np.mean(self.scores[-20:])<200: break
    
    def test(self):
        state = self.env.reset(seed=self.random_seed)[0]
        done = False 
        while not done: 
            action = self.action_choice(state)
            update_state,reward,terminated,truncated,_ = self.env.step(action)
            done = terminated or truncated
            np.save(self.temp_folder+'env_rgb',self.env.render())
            with open(self.temp_folder+'env_info.txt','w') as f:   
                f.write('%.3f | %.3f | %.3f | Episode: %d | Step: %d'
                        %(update_state[0],update_state[1],reward,self.env._elapsed_steps))
        
 

    
        
            
    def save_model(self, file_name): self.model.save(file_name) 
    def load_model(self, file_name): self.model = tf.keras.models.load_model(file_name)
    
    def score_plot(self):
        plt.plot(range(1,len(self.scores)+1),self.scores)
        plt.title('DQN scores')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.grid()
        plt.show()

        plt.plot(range(1,len(self.losses_list)+1),self.losses_list)
        plt.title('DQN losses')
        plt.ylabel('Loss')
        plt.xlabel('Episode')
        plt.grid()
        plt.show()