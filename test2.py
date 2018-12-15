import pandas as pd
df = pd.read_csv("test/test.csv")

import numpy as np
import matplotlib.pyplot as plt

prices = np.array(df['price'])
actions = np.array(df['action'])

def test(actions, prices):
    usd = 1000.0
    btc = 0.0
    current = 1000.0
    history = []
    for i in range(len(prices)-1):
        if actions[i] == 0:
            if usd/current > 0.1:
                buy_money=usd-current*0.1
                buy_btc = buy_money/prices[i]
                usd-=buy_money
                btc+=buy_btc
        elif usd/current < 0.9:
            sell_money=current*0.9-usd
            sell_btc = sell_money/prices[i]
            usd+=sell_money
            btc-=sell_btc
        current = usd + btc*prices[i]
        history.append(current/1000.0 - 1)
    return history

optimal = []
for i in range(len(prices)-1):
    if prices[i] < prices[i+1]:
        optimal.append(0)
    else:
        optimal.append(1)

history_learn = test(actions, prices)
history_opt = test(optimal, prices)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(aspect = 2000)
plt.plot(history_learn)
#plt.plot(history_opt)
plt.plot(prices/prices[0] - 1)
#plt.savefig('test/result_1_opt.png')
#plt.plot(actions, 'bs')
#plt.savefig('test/actions_1.png')
plt.show()