import numpy as np 
import matplotlib.pyplot as plt 
import time

ENV_RENDER_FILE='../temp/env_rgb.npy'
ENV_INFO_FILE='../temp/env_info.txt'

old_time = 0 
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')
axim = ax.imshow(np.load(ENV_RENDER_FILE))
with open(ENV_INFO_FILE,'r') as f: env_info = f.readline() 
ax.set_title(env_info)

while True:
    if (time.time() - old_time) > 0.1:
        old_time = time.time()
        try: 
            axim.set_data(np.load(ENV_RENDER_FILE))
        except Exception as e: print(e)
        with open(ENV_INFO_FILE,'r') as f: env_info = f.readline()
        ax.set_title(env_info)
        fig.canvas.flush_events()


