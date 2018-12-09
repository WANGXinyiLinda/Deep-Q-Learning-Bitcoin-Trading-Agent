# Deep Q Learning Bitcoin Trading Agent

## Q-network:

    map both state and action to a single Q value
    
    structure: CONV(32) - CONV(64) - GRU(128) - GRU(128) - Dense(64) - Dense(32)

## Memory replay:

    use the optimal policy instead of the past experience. (details see SCIE3500_Final_Report.pdf, Section 5.3)

## Usage:

    Train: python train.py
    
    Test: python test.py, python test2.py
    
    Tune parameters: Change values in constants.py