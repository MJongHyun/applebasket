from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
# inertia 계산 : KMeans 계산

def inertia(data, column):
    k = data[column]
    cn = range(1, 20)
    res = []
    for i in cn:
        model = KMeans(n_clusters=i)
        model.fit(k)
        res.append(model.inertia_)

    plt.plot(cn, res, '-o')
    plt.xlabel('num of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(cn)
    plt.show()


# clustering meanshift

def meanshift(data, columns, width): # 데이터, 컬럼, meanshift 높이 기준

    test = data[columns]
    clustering = MeanShift(bandwidth=width).fit(test)
    data['predict'] = clustering.labels_

    return data

# clustering Kmeans

def kmean(data, k, columns):
    test = data[columns]
    model = KMeans(n_clusters=k)
    pre = model.fit(test)
    data['predict'] = pre.labels_

    return data

# standardScaler, kmeans 모델

def cluster_model(data, k, columns):
    test = data[columns]
    scaler = StandardScaler()  # 값이 평균 0과 표준편차 1이 되도록 변환
    model = KMeans(n_clusters=k)  # Kmeans clustering
    pipeline = make_pipeline(scaler, model)  # pipeline package를 통해 scaler 모델과 k-means 모델 동시 수행
    pipeline.fit(test)  # data를 pipeline에 적용

    predict = pd.DataFrame(pipeline.predict(test))
    predict.columns = ['predict']

    data = data.reset_index(drop=True)
    data = pd.concat([data, predict], axis=1)

    return data