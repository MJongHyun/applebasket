# 필요데이터 : allresult_data (주류,로컬,소상공인데이터로 구성되어있다.)

import pandas as pd
from scipy.stats import boxcox
from sklearn.cluster import KMeans
from sklearn import svm as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 변화율 함수

def min_max(data, col, num): # 입력값 : 지수관련데이터, 지수컬럼, min_max화 할 값

    return num * ((data[col] - data[col].min())/(data[col].max() - data[col].min()))

# 함수 설명

"""각 지수에 해당하는 가장 작은 값을 해당하는 지수로 나눈 후 반올림 (동점자 추출)
1에서 계산한 값을 바탕으로 rank를 진행하여 동점자는 평균등수로 나올 수 있게 진행한 후, 반올림 (동점자 추출)
모든 지수의 Scale을 같게하기 위해 Min_Max_Scale을 사용하여 모든 지수 값의 범위를 일정하게 함

다양성의 경우, 위의 결과와 같이 3등분으로 나오기 때문에 scipy 패키지에서 정규화를 시켜 min_max 후 위와 같은
설명방식의 함수를 적용하여 진행"""

def rank_min_max(data, col, num):  # 입력값 : 지수관련데이터, 지수컬럼, min_max화 할 값

    df = data.copy()

    if df[col].min() == 0:  # 최소값이 0인 경우, 값을 나눌 수 없기 때문에 1로 바꿈
        m = 1
    else:
        m = df[col].min()

    df['I'] = round(df[col] / m, 0)  # 설명 1번
    df['R'] = round(df.I.rank(), 0)  # 설명 2번
    res = min_max(df, 'R', num)  # 설명 3번

    return res

def ent_min_max(data,col,num): # 입력값 : 다양성지수 관련 데이터, 다양성 컬럼, min_max화 할 값

    normal = pd.DataFrame(boxcox(data[col])[0])
    normal.columns = ['정규화']
    normal['변환'] =  min_max(normal,'정규화' ,num)
    normal['다양성'] = rank_min_max(normal,'변환' ,num)

    return normal['다양성']

def change_rate(name_col,start,end,data): # 입력데이터: 상권정보컬럼, 소상공인-로컬데이터 시작, 끝, 소상공인-로컬데이터

    name = data[name_col]
    target = data.loc[:,start:end]
    target['sum'] = target.sum(axis = 1)

    res = pd.concat([name,target], axis = 1)

    ch_data = pd.DataFrame()

    for i in res.COMMERCIAL_ID.unique():

        ch_tar = res[res.COMMERCIAL_ID == i]['sum']
        min_max = (ch_tar.max() - ch_tar.min()) / ch_tar.max()

        ch_d = pd.DataFrame([i, min_max]).T

        ch_d.columns = ['COMMERCIAL_ID', 'NEW_CH']
        ch_data = pd.concat([ch_data, ch_d], axis=0)

    total = pd.merge(name,ch_data)
    total = pd.merge(total,res[['YM','COMMERCIAL_ID','sum']])

    return total

# 밀집도 함수 - 위에서 구한 데이터와 지형데이터를 합쳣다고 가정

def area_p(data):

    return data['sum']/data['AREA']

# clustering Kmeans

def kmean(data, k, columns): # 입력데이터 : 지수관련데이터, 군집갯수, 군집할 컬럼

    test = data[columns] # 군집화 할 값 설정
    model = KMeans(n_clusters=k) # 군집 갯수 지정 후 KMeans 실행

    pre = model.fit(test) # 모델 적용
    data['predict'] = pre.labels_ # 군집화 값 추출

    return data

# 지도학습(SVM)

def acc_model(data, jisu_col, pre): # 입력데이터 : 군집데이터, 군집필요컬럼, 군집번호컬럼

    X_train, X_test, y_train, y_test = train_test_split(data[jisu_col], data[pre], test_size=0.3)

    model = ss.SVC(kernel='linear', gamma=0.01, C=0.1)
    model.fit(X_train, y_train)

    random_best_train = model.predict(X_train)
    random_best_test = model.predict(X_test)

    print("정확도 1 : ", accuracyscore(y_train, random_best_train))
    print("정확도 2 : ", accuracy_score(y_test, random_best_test))

    return model

# 군집화 관련 레이더 차트 그리기

def total_graph(x, y, cn, data, column, des, marker = [0,2,4,6,8,10]): # 입력값 : 가로, 세로, 그래프 수, 데이터, 그래프에 나타낼 컬럼, 표시 기준리스트

    # 라벨과 makers 설정
    labels = column
    markers = marker

    # 시작점 지정 및 그래프에 나타낼 컬럼 지정
    pre = data[data['상권군집번호'] == 0] # predict
    graph = pre[column].describe().loc[des]

    # 라벨 갯수에 따른 원형 각도를 구하고, 각각 위치 값들 설정
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((graph, [graph[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # 첫번째 그래프 추출
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(x, y, 1, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.2)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    plt.yticks(markers)
    ax.grid(True)

    # 나머지 그래프 추출
    for i in range(1, cn):
        pre = data[data['상권군집번호'] == i] # predict
        graph = pre[column].describe().loc[des]

        ax = fig.add_subplot(x, y, i + 1, polar=True)
        ax.plot(angles, np.concatenate((graph, [graph[0]])), 'o-', linewidth=2)
        ax.fill(angles, np.concatenate((graph, [graph[0]])), alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.yticks(markers)
        ax.grid(True)


# 군집그래프 겹쳐서 비교

def ind_graph(data, column, des, entry=False, marker = [0,2,4,6,8,10]): # 입력 값: 군집데이터, 컬럼, 표시 기준, 그래프 기준점

    if not entry:
        entry = data.predict.unique()
    else:
        entry = entry

    labels = column
    markers = marker

    ax = plt.subplot(1, 1, 1, polar=True)

    # entry에 있는 값에 따라 군집 겹쳐서 그래프 추출

    for i in entry:

        pre = data[data['predict'] == i]
        graph = pre[column].describe().loc[des]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        stats = np.concatenate((graph, [graph[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax.plot(angles, stats, 'o-', linewidth=2, label=i)
        ax.fill(angles, stats, alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.yticks(markers)
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))