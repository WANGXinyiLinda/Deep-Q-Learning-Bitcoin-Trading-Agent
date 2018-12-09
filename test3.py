import pandas as pd
df = pd.read_csv("logs/history.csv")
loss = df['loss']

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(aspect = 100)
plt.plot(loss)
#plt.plot(actions)
plt.savefig('logs/loss.png')
plt.show()