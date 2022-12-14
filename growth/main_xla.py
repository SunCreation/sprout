import numpy as np
import torch as th
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, GRU
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse  as ap
from utils import (
    incloud_null_col,
    predict_columns,  
    preprocess_remove_str,
    slice_sample,
    remove_outlier,
    dummy_onehot_concat,
    CosineSchedule
)
from src import (
    Sprout_Dense,
    train
)
import yaml
import wandb
from wandb.keras import WandbCallback 



resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

perser = ap.ArgumentParser()
perser.add_argument('-d','--data_path',type=str, default='./data')
perser.add_argument('-c','--config',type=str, default='./config/default.yaml')

# perser.add_argument('-t','--train_model',type=str,default='./src/train.py')

args =  perser.parse_args()

data_path = args.data_path

namedict = {
    '개화군':'flower_num',
    '내부CO2': 'co2',
    '내부습도': 'humidity',
    '내부온도': 'in_temp',
}


def train():

    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    wandb.init(config=config,project="sprout", entity="suncreation",name=f'val0001_{config["epochs"]}epochs_interpolation(hum,tem)_incloud(outdata)_preprocess(output-flower)_batch{config["batchsize"]}')
    config = wandb.config
    # 데이터 불러오기
    inputs = pd.read_csv(f'{data_path}/train_input.csv')
    outputs = pd.read_csv(f'{data_path}/train_output.csv')

    inputs_ = inputs

    # print(inputs[incloud_null_col(inputs)[0]])
    # remove_outlier(inputs)


    inputs = dummy_onehot_concat(inputs, ['재배형태'])
    # inputs = pd.concat([inputs,pd.get_dummies(inputs['재배형태'])],axis=1)
    # inputs = pd.concat([inputs,pd.get_dummies(inputs['시설ID']),pd.get_dummies(inputs['재배형태'])],axis=1)

    del inputs['재배형태']
    # del inputs['외부온도']
    # del inputs['외부풍향']
    # del inputs['외부풍속']
    del inputs['품종']
    # del inputs['지습']

    # del inputs['일']
    # del inputs['Sample_no']

    # str데이터 전처리
    inputs = inputs.apply(preprocess_remove_str)

    # sample_no, 시설 ID, 날짜값 제외
    print(inputs.iloc[:,3:])

    # 이상치 전처리
    remove_outlier(inputs, outputs)

    # scaling
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    input_sc = input_scaler.fit_transform(inputs.iloc[:,3:].to_numpy())
    output_sc = output_scaler.fit_transform(outputs.iloc[:,3:].to_numpy())

    print(input_sc)

    # sample 별로 데이터 종합
    input_ts = slice_sample(inputs, outputs, input_sc)

    print(input_ts.shape)
    input_ts = np.where(tf.math.is_nan(input_ts), 0, input_ts)

    # 셋 분리
    train_x, val_x, train_y, val_y = train_test_split(input_ts, output_sc, test_size=0.001,
                                                    shuffle=True, random_state=0)


    # # 모델 정의
    # def create_model():
    #     inputs_ = Input(shape=[7, 13])
    #     x = inputs_
    #     x = LSTM(512, return_sequences=True)(x)
    #     l1 = LSTM(512)(x)
    #     out = Dense(3, activation='relu')(l1+x[:,-1])
    #     return Model(inputs=inputs_, outputs=out)

    # 모델 정의
    def create_model(num, input_shape):
        input_ = Input(shape=input_shape)
        x = Dense(num)(input_)
        x = LSTM(num,return_sequences=True)(x)
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        x = Dense(num)(x + resi)
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        x = x + resi
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        x = x + resi
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        x = Dense(num)(x + resi)
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        x = x + resi
        resi = x
        x = LSTM(num,return_sequences=True)(x)
        l1 = LSTM(num)(x + resi)
        out = Dense(3, activation='tanh')(l1)
        return Model(inputs=input_, outputs=out)

    EPOCHS = config.epochs
    BATCH_SIZE = config.batchsize
    learning_rate = CosineSchedule(train_steps=EPOCHS*input_ts.shape[0]//BATCH_SIZE,offset=3e-4,decay=1e-4)

    with strategy.scope():
        model = create_model(config.hiddensize, input_ts.shape[1:])
        # checkpointer = ModelCheckpoint(monitor='val_mae', filepath='baseline.tf',
        #                             verbose=1, save_best_only=True, save_weights_only=True)
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),metrics=config.metrics)

    model.summary()

    # 학습
    hist = model.fit(train_x, train_y, 
                     batch_size=BATCH_SIZE, 
                     epochs=EPOCHS, 
                     validation_data=(val_x, val_y), 
                     callbacks=[WandbCallback(data_type = 'csv',
                                training_data = (val_x, val_y)
                                ,save_model=False)])


    # loss 히스토리 확인
    # plt.figure(figsize=(20,15))
    # fig, loss_ax = plt.subplots(nrows=2)
    # loss_ax[0].plot(hist.history['loss'], 'r', label='loss')
    # loss_ax[0].plot(hist.history['val_loss'], 'g', label='val_loss')
    # loss_ax[0].set_xlabel('epoch')
    # loss_ax[0].set_ylabel('loss')
    # loss_ax[0].legend()
    # loss_ax[0].set_title('loss - val loss plot')
    # loss_ax[0].set_ylim(0,0.005)
    # loss_ax[1].plot(hist.history['mae'], 'r', label='mae')
    # loss_ax[1].plot(hist.history['val_mae'], 'g', label='val_mae')
    # loss_ax[1].set_xlabel('epoch')
    # loss_ax[1].set_ylabel('mae')
    # loss_ax[1].legend()
    # loss_ax[1].set_title('mae - val mae plot')
    # loss_ax[1].set_ylim(0,0.04)
    # plt.savefig('now.png')


    # 저장된 가중치 불러오기
    # model.load_weights('baseline.h5')

    def postprocess(df, cols):
        for col in cols:
            df[col] = df[col].apply(lambda x: max(x,0))
        
    # 테스트셋 전처리 및 추론
    test_inputs = pd.read_csv(f'{data_path}/test_input.csv')
    output_sample = pd.read_csv(f'{data_path}/answer_sample.csv')

    remove_outlier(test_inputs)

    test_inputs = test_inputs[inputs_.columns]

    test_inputs = dummy_onehot_concat(test_inputs, ['재배형태'])
    # test_inputs = pd.concat([test_inputs,pd.get_dummies(test_inputs['품종']),pd.get_dummies(test_inputs['재배형태'])],axis=1)
    # test_inputs = pd.concat([test_inputs,pd.get_dummies(test_inputs['시설ID']),pd.get_dummies(test_inputs['재배형태'])],axis=1)

    del test_inputs['재배형태']
    # del test_inputs['외부온도']
    # del test_inputs['외부풍향']
    # del test_inputs['외부풍속']
    del test_inputs['품종']
    # del test_inputs['지습']

    # del test_inputs['재배형태']

    test_inputs = test_inputs.apply(preprocess_remove_str)

    test_input_sc = input_scaler.transform(test_inputs.iloc[:,3:].to_numpy())

    test_input_ts = slice_sample(test_inputs, output_sample, test_input_sc)

    test_input_ts = np.where(tf.math.is_nan(test_input_ts), 0, test_input_ts)

    prediction = model.predict(test_input_ts)

    prediction = output_scaler.inverse_transform(prediction)
    output_sample[['생장길이', '줄기직경', '개화군']] = prediction
    postprocess(output_sample, ['생장길이', '줄기직경', '개화군'])

    # 제출할 추론 결과 저장
    output_sample.to_csv(f'../pre/val0001_{config.epochs}epochs_interpolation(hum,tem)_incloud(outdata)_preprocess(output-flower)_batch{config.batchsize}.csv', index=False)


if __name__=='__main__':
    wandb.login()
    with open('config/sweep.yaml') as f:
        sweep_config = yaml.load(f,Loader=yaml.FullLoader)
    # sweep_id = wandb.sweep(sweep_config)

    # run the sweep
    wandb.agent('i3qvz0n9',
                function=train,
                project="sprout", 
                entity="suncreation")