import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import matplotlib.style as style
style.use('fivethirtyeight')

from matplotlib import rc

rc('font', family='AppleGothic')
rc('axes', unicode_minus=False)

from scipy.stats import skew

# 목적 : 선형회귀를 진행하기전 컬럼의 분포 및 정규성, 특잇값을 파악하기 위해 컬럼의 특징을 살펴봄

def plotting_3_chart(df, feature):  # 입력값 : 데이터, 데이터컬럼

    # 차트 생성
    fig = plt.figure(constrained_layout = True, figsize = (15, 10))
    grid = gridspec.GridSpec(ncols = 3, nrows = 3, figure = fig)

    # 히스토그램 생성
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:, feature], norm_hist = True, ax = ax1)

    # QQplot 생성
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:, feature], plot = ax2)

    # boxplot 생성
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:, feature], orient = 'v', ax = ax3);


def col_dtype(df):  # 입력값 : 데이터

    # 컬럼형태가 numeric인지 아닌지 파악

    numeric_col = df.dtypes[df.dtypes != "object"].index
    ob_col = df.dtypes[df.dtypes == "object"].index

    return numeric_col, ob_col


# 데이터에서 Null값 확인

def null_df(df):  # 입력값 : 데이터

    null_sum = df.isnull().sum().reset_index(drop=False)
    null_sum.columns = ['Column', 'Null_cn']

    null_df = null_sum[null_sum['Null_cn'] != 0].sort_values('Null_cn')
    null_df['P'] = null_df['Null_cn'] / len(df)

    return null_df


# 상관관계 파악 - 종속변수와 상관관계가 높은 값들을 추출

def corr_df(df, res_name, k, size = 15):  # 입력값 : 데이터, 종속변수컬럼이름, 컬럼 값 뽑을 갯수

    fig, ax = plt.subplots(figsize=(size, size))
    corrmat = df.corr()

    cols = corrmat.nlargest(k, res_name)[res_name].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.2, font = 'AppleGothic')
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


# 종속변수와의 선형성 및 등분산성 확인을 하는 그래프

def li_Hom_check(data, indep_col, dep_col):

    # 그래프크기 결정
    fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), ncols=2, sharey=False)

    # 산포도 확인
    sns.scatterplot(x=data[dep_col], y=data[indep_col], ax=ax1)

    # 회귀값 확인
    sns.regplot(x=data[dep_col], y=data[indep_col], ax=ax1)

    # 등분산성 확인
    sns.residplot(x=data[dep_col], y=data[indep_col], ax=ax2)

# skewness 확인하는 그래프

def skew_check(data, col):  # 입력값 : 데이터, skewness 확인컬럼

    return data[col].apply(lambda x: skew(x)).sort_values(ascending=False)

