import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()


def incloud_null_col(df: pd.DataFrame):
    '''
    return ({not incloud null data}, {incloud null data}) 
    '''
    columns = df.columns
    return columns[df.isnull().sum()==0], columns[df.isnull().sum()!=0]

def preprocess_remove_str(df:pd.Series) -> pd.Series:
    '''
    def preprocess_remove_str(df:pd.Series) -> pd.Series:
        return pd.Series(np.vectorize(lambda x: np.float(re.sub('[^\d]','',x)))(df))
    '''
    def to_number(x):
        if type(x) is str:
            if re.sub('[^\d]','',x):

                return np.float(re.sub('[^\d]','',x))
        
        return x
    return df.apply(to_number) # pd.Series(np.vectorize(to_number)(df))



# 입력 시계열화
def slice_sample(inputs, outputs, input_sc):
    input_ts = []
    for i in outputs['Sample_no']:
        sample = input_sc[inputs['Sample_no'] == i]
        if len(sample < 7):
            sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample,
                            axis=0)
        sample = np.expand_dims(sample, axis=0)
        input_ts.append(sample)
    input_ts = np.concatenate(input_ts, axis=0)

    return input_ts


def remove_outlier(df: pd.DataFrame):
    df['내부CO2'] = np.vectorize(lambda x: x*100 if x < 10 else x)(df['내부CO2'])
    df['내부온도'] = np.vectorize(lambda x: x*10 if x < 10 else x)(df['내부온도'])
    df['내부습도'] = np.vectorize(lambda x: x*10 if x < 10 else x)(df['내부습도'])
    interpolate(df, len(df), ('내부CO2','내부온도','내부습도'))



def interpolate(seg, length, cols):
    for col in cols:
        count = 0
        result = 0

        for i in range(length):
            if pd.isna(seg[col][i]):
                continue
            if seg[col][i] != 0:
                count += 1
                result += seg[col][i]

            output = result/count

        for i in range(length):
            if pd.isna(seg[col][i]) or seg[col][i] == 0:
                seg[col][i] = output
