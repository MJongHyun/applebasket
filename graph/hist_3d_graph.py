
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# heatmap 생성

def heatmap(data,column):

    # 데이터 히스토그램 할 값 설정
    his = data[column]

    # correlation 계산
    cor = his.corr()

    # 그래프 생성
    plt.figure(figsize=(15, 15))
    sns.heatmap(data=cor, annot=True, fmt='.2f', linewidths=5, cmap='Blues')
   plt.figure(figsize=(20,10))

    for i in range(0,len(columns)):

        name = columns[i]

        plt.subplot(x, y, i + 1)
        plt.hist(data[name], histtype='bar', rwidth=0.9, bins=bin)
        plt.title(name)

    plt.show()
# 히스토그램 생성

def histogram(data,columns,x,y,bin=10): # 데이터, 컬럼, 그래프 갯수 ex) 224



# 3d scatter 그래프 생성

def plot_3d(data,col1,col2,col3,div): # 데이터, 컬럼명들, 구분컬럼, 그래프사이즈 지정

    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(data[col1], data[col2], data[col3], c=data[div], alpha = 1)

# 히스토그램 군집화 함수

def plot_corr(df, size=10):

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='Blues')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);

    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)


def cluster_corr(A, size):

    X = A.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [A.columns.tolist()[i] for i in list((np.argsort(ind)))]
    A = A[columns]
    plot_corr(A, size)


