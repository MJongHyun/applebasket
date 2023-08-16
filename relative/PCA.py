from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 스크리그래프 : 성분들의 변동을 표기한그림, 성분들의 상대적인 중요도를 보여주는 그래프

def screeplot(data, data_column, n_comp): # PCA 진행할 데이터, 데이터컬럼, 차원축소 컬럼 수

    # 데이터 scale화
    scaler = StandardScaler()
    scaler.fit(data)
    ath_scale = scaler.transform(data)

    # PCA 진행
    pca = PCA(n_components=n_comp)
    pca.fit(ath_scale)

    # 상대적인 값 추출
    percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    end = len(percent_variance)+1

    # 위에 값 그래프 화 (값이 클수록, 변동의 크다는 것을 나타냄)
    plt.bar(x=range(1, end), height=percent_variance, tick_label=data_column[:n_comp])
    plt.xticks(rotation=45)
    plt.show()

    return pca

# PCA를 통해 최적화된 성분 갯수 정하기

def best_pca_cn(X,Y,n_comp): # 군집할 데이터, 결과데이터, 성분 갯수

    result = pd.DataFrame()

    for num in range(2,n_comp+1):

        # train/test set을 나눔
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

        # pca 적용
        pca = RandomizedPCA(n_components=num, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # GridSearchCV로 최적화된 모델 추출
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

        clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, y_train)

        # 모델을 통해 정확도를 측정하여 정확도가 가장 높은 값을 모델결정 및, 성분 갯수 결정

        random_best_train = clf.predict(X_train_pca)
        random_best_test = clf.predict(X_test_pca)

        train_acc = accuracy_score(y_train, random_best_train)
        test_acc = accuracy_score(y_test, random_best_test)

        G = pd.DataFrame([num, train_acc, test_acc]).T
        G.columns = ['N_COMP', 'TRAIN_ACC', 'TEST_ACC']

        result = pd.concat([result, G], axis = 0)

    return result
