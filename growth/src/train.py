import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import torch as th
from torch import nn
from torch import functional as F
# null값이 있는  데이터와 없는 데이터를 따로 학습시켜서 
# 없는 데이터는 그에 걸맞게 학습이 될  수 있게 만든다. 마스크 사용




def train():
    ...