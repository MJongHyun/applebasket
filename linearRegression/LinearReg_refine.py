import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# 로그화하여 정규성으로 만들기

def log_norm(df, col): #입력값 : 데이터, log값을 적용할 column

    return np.log1p(df[col])

# boxcox의 전제조건 : 모든 값이 0보다 커야한다. 음수도 해야 할 경우 여-존슨 변환Yeo-Johnson Transformation를 검색.

def fixing_skewness(df, skew_value):  # 입력값 : 데이터, 정규화 시킬 skew값 결정

    # 수치데이터 컬럼만 추출
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # 수치데이터의 skew값을 추출 후, skew값이 일정이상 값을 초과하는 feature을 추출
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > skew_value]
    skewed_features = high_skew.index

    # 위에서 추출한 feature들을 정규화시키기 위해 boxcox 방법을 사용하여 정규화된 값으로 추출
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

    return df

# 데이터 값에서, 너무 하나의 값으로 치중되어 있는 경우 ((예) 100개의 데이터 중 0이 99개, 1이 1개인 경우), 데이터를 확인하기 위해 컬럼추출

def overfit_reducer(df, overfit_rate):  # 입력값 : 데이터, overfit으로 결정할 값

    # overfit값 추출
    overfit = []

    # 컬럼 중 하나의 값으로 치우친 값이 있는 컬럼의 경우, overfit_list에 저장
    for i in df.columns:

        counts = df[i].value_counts()  # 컬럼 안에서 각 값의 갯수 추출 (높은 값부터 추출됨)
        zeros = counts.iloc[0]  # 해당컬럼에서 가장 많은 값

        value_rate = zeros / len(df)  # 비율 값 추출

        if value_rate * 100 > overfit_rate:
            overfit.append(i)

    overfit = list(overfit)

    return overfit


# 범주형 데이터 dummie화

def dummie_value(df, column):  # 입력값: 데이터, 더미화에 필요한 컬럼

    # 해당 컬럼들을 더미화 한 후, 더미화를 진행하였을때, 사용한 컬럼은 제거
    dummie = pd.get_dummies(data[column]).reset_index(drop=True)
    df = df.drop(column, axis='columns').reset_index(drop=True)

    data = pd.concat([df, dummie], axis=1)

    return data


# 연속형데이터를 이산형데이터로 바꾸기

def discretization_value(df, bins, sel_col):  # 입력값: 데이터, 구간, 이산형으로 바꿀 데이터

    # 연속형 데이터의 최대/최소를 구한 후, 구간을 정하여 구간 만큼 나눈다.
    bin_ = np.linspace(df[sel_col].min(), df[sel_col].max(), bins)

    # 나눈 값에 대하여, 최소값을 1로하여 최대값까지 구간을 나눈 만큼 이산화하여 바꾼다.
    disc_val = np.digitize(df[sel_col], bin_)

    return disc_val