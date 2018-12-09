# Deep Q Learning Bitcoin Trading Agent

This is a continued work of the RIPS-HK 2018 group project, in which our emphasis is more of understanding the mechanism of Bitcoin and reproducing previous works. In this project, I focus more on try different ways to implement the deep Q-learning algorithm. I used two kind of neural networks: ResNet and recurrent neural network (RNN). I tried two approaches to implement the Q-network: the rst one is to use the Q-network to map the state to the Q-values of all actions, the other one is to use the Q-network to map both the state and a specic action to a single Q-value. It turns out that the latter scheme with RNN has a better performance.

![](doc-image/deep-q-network-example.png)

## Action space:

    {BUY, SELL}


## State:

![](doc-image/State.png)

## Q-network:

    map both state and action to a single Q value
    
    structure: CONV(32) - CONV(64) - GRU(128) - GRU(128) - Dense(64) - Dense(32)

## Memory replay:

    use the optimal policy instead of the past experience. (details see SCIE3500_Final_Report.pdf, Section 5.3)

## Usage:

    Train: python train.py
    
    Test: python test.py, python test2.py
    
    Tune parameters: Change values in constants.py

## Test result:

![](doc-image/result32.png)

The orange curve is the percentage price change comparing to the initial price. And the blue curve is the percentage change of the asset value comparing to the initial asset.