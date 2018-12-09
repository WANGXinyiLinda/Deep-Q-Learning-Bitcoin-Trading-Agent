from agent import Agent
from processor import Processor
import pandas as pd

processor = Processor()
agent = Agent(processor)
agent.test()