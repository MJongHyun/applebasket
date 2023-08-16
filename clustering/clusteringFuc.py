from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm as ss


def cluster_model(data, k):
    scaler = StandardScaler()  # 값이 평균 0과 표준편차 1이 되도록 변환
    model = KMeans(n_clusters=k)  # Kmeans clustering
    pipeline = make_pipeline(scaler, model)  # pipeline package를 통해 scaler 모델과 k-means 모델 동시 수행
    pipeline.fit(data)  # data를 pipeline에 적용

    predict = pd.DataFrame(pipeline.predict(data))
    predict.columns = ['predict']
    return predict


def clustering_balance(data, cluster_date):  # balance 관련 군집화

    target_data = data[data['WR_DT'] == cluster_date]  # cluster_date에 해당하는 월데이터 추출
    # balance에 따른 클러스터링을 구분하기 위해 하나의 축을 고정하고, 나머지 하나측을 BALANCE 값으로 하여 데이터를 추출
    target_bal = pd.DataFrame([np.array([1 for i in range(0, len(target_data))]), target_data['BALANCE']]).T
    cluster_bal = cluster_model(target_bal, 2)  # cluster model을 통해 2가지 분류로 나눔

    # model값을 통해 나온 값과 기존에 있는 값 데이터를 합침
    target_data = target_data.reset_index(drop=True)
    cluster_bal = cluster_bal.reset_index(drop=True)

    total_bal = pd.concat([target_data, cluster_bal], axis=1)

    return total_bal


def clustering_exp(total_bal):  # 나머지 지수 군집화

    # 위에서 balance로 군집화한 값을 바탕으로 2가지 종류로 데이터를 나눔
    total_bal_0 = total_bal[total_bal['predict'] == 0].iloc[:, :-1]
    total_bal_1 = total_bal[total_bal['predict'] == 1].iloc[:, :-1]

    # 데이터에서 군집화에 필요한 3가지 지수 값들 추출
    test_0 = total_bal_0[['PLEASURE', 'LIFE', 'POPULATION']]
    test_1 = total_bal_1[['PLEASURE', 'LIFE', 'POPULATION']]

    # cluster_model을 통해 각 데이터당 5개의 군집화 추출
    cluster_test_0 = cluster_model(test_0, 5)
    cluster_test_1 = cluster_model(test_1, 5)

    # 각각 군집화 관련된 데이터들을 합침
    total_bal_0 = total_bal_0.reset_index(drop=True)
    cluster_test_0 = cluster_test_0.reset_index(drop=True)

    total_bal_1 = total_bal_1.reset_index(drop=True)
    cluster_test_1 = cluster_test_1.reset_index(drop=True)

    target_0 = pd.concat([total_bal_0, cluster_test_0], axis=1)
    target_1 = pd.concat([total_bal_1, cluster_test_1], axis=1)

    # 군집화하여 합친 각각 데이터 2개를 비교하여 BALANCE가 큰데이터를 target_0, 작은데이터를 target_1으로 설정

    if target_0['BALANCE'].mean() > target_1['BALANCE'].mean():
        target_1['predict'] = target_1['predict'] + 5  # BALANCE가 작은 값들로 구성된 군집화 값을 구분하기 위해 5를 더함

    elif target_0['BALANCE'].mean() < target_1['BALANCE'].mean():

        temp = target_0
        target_0 = target_1
        target_1 = temp
        target_1['predict'] = target_1['predict'] + 5

    target_total = pd.concat([target_0, target_1], axis=0)

    return target_0, target_1, target_total


def opt_model(target_0, target_1):  # randomizeSearchCV 를 통해 svm model을 최적화 함

    # SVM_Model - 데이터를 선형으로 분리하는 최적의 선형 결정 경계를 찾는 알고리즘

    C = [0.1, 1, 10, 100, 1000]  # 오분류값을 강하게 줄지 말지 결정 (오분류 : 제대로 예측하지 못한 관측지를 평가하는 지표)
    kernel = ['rbf', 'linear']  # 비선형, 선형 - rbf: 비선형 mapping을 통해 고차원으로 변환 후, 새로운 차원에서 최적의 결정 경계면 찾기
    gamma = [0.01, 0.1, 1, 10, 100]  # 선형 - 비선형 평면으로 나누는 것을 결정하는 parameter
    random_svm = {'C': C, 'kernel': kernel, 'gamma': gamma}

    svm = ss.SVC()
    svm.probability = True

    # 최적화 모델 parameter 찾기 - randomizedsearchCV : parameter의 각기 다른 값들을 탐색하며 무작위 조합에 대해 평가
    # estimator : 최적화 대상모델, param_distributions : 대상 모델의 parameter 값들, n_iter : 시도 횟수
    # cv : fold의 수 (교차검증 갯수) , verbose: 추출되는 문장 수 , random_state: sampling 뽑을 때 사용하는 난수 생성기
    # n_jobs : cpu 코어의 수 (-1 인 경우 전체 사용)
    svm_opt_0 = RandomizedSearchCV(estimator=svm, param_distributions=random_svm, n_iter=50, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    svm_opt_1 = RandomizedSearchCV(estimator=svm, param_distributions=random_svm, n_iter=50, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # RandomizedSearchCV 를 통해 얻은 최적화 모델에 각각 데이터를 적용
    svm_opt_0.fit(target_0[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']], target_0['predict'])
    svm_opt_1.fit(target_1[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']], target_1['predict'])

    return svm_opt_0, svm_opt_1


from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm as ss


def cluster_model2(data, k):
    scaler = StandardScaler()  # 값이 평균 0과 표준편차 1이 되도록 변환
    model = KMeans(n_clusters=k)  # Kmeans clustering
    pipeline = make_pipeline(scaler, model)  # pipeline package를 통해 scaler 모델과 k-means 모델 동시 수행
    pipeline.fit(data)  # data를 pipeline에 적용

    predict = pd.DataFrame(pipeline.predict(data))
    predict.columns = ['predict']
    return predict


def clustering_balance(data, cluster_date):  # balance 관련 군집화

    target_data = data[data['WR_DT'] != cluster_date]  # cluster_date에 해당하는 월데이터 추출
    # balance에 따른 클러스터링을 구분하기 위해 하나의 축을 고정하고, 나머지 하나측을 BALANCE 값으로 하여 데이터를 추출
    target_bal = pd.DataFrame([np.array([1 for i in range(0, len(target_data))]), target_data['BALANCE']]).T
    cluster_bal = cluster_model(target_bal, 2)  # cluster model을 통해 2가지 분류로 나눔

    # model값을 통해 나온 값과 기존에 있는 값 데이터를 합침
    target_data = target_data.reset_index(drop=True)
    cluster_bal = cluster_bal.reset_index(drop=True)

    total_bal = pd.concat([target_data, cluster_bal], axis=1)

    return total_bal


# target_modeling을 통해 나온 최적화된 모델에 각각 데이터를 적용
def total_modeling(data, cluster_date, svm_opt_0, svm_opt_1, target_total):
    # balance를 기준으로 cluster_model를 통해 두개의 군집화로 나눔
    total_bal = clustering_balance(data, cluster_date)

    total_bal_0 = total_bal[total_bal['predict'] == 0].iloc[:, :-1]
    total_bal_1 = total_bal[total_bal['predict'] == 1].iloc[:, :-1]

    # 군집화하여 합친 각각 데이터 2개를 비교하여 BALANCE가 큰데이터를 target_0, 작은데이터를 target_1으로 설정

    if total_bal_0['BALANCE'].mean() > total_bal_1['BALANCE'].mean():

        opt_0 = svm_opt_0.best_estimator_.predict(total_bal_0[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']])
        opt_1 = svm_opt_1.best_estimator_.predict(total_bal_1[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']])

    elif total_bal_0['BALANCE'].mean() < total_bal_1['BALANCE'].mean():
        temp = total_bal_0
        total_bal_0 = total_bal_1
        total_bal_1 = temp

    # 각각 나눈 데이터를 taget_modeling에서 나온 모델을 적용하여 군집화 값 추출

    opt_0 = svm_opt_0.best_estimator_.predict(total_bal_0[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']])
    opt_1 = svm_opt_1.best_estimator_.predict(total_bal_1[['PLEASURE', 'LIFE', 'BALANCE', 'POPULATION']])

    opt_0 = pd.DataFrame(opt_0, columns=['predict'])
    opt_1 = pd.DataFrame(opt_1, columns=['predict'])

    # 군집화하여 나온 결과값과 데이터들을 합침

    total_bal_0 = total_bal_0.reset_index(drop=True)
    total_bal_1 = total_bal_1.reset_index(drop=True)

    final_0 = pd.concat([total_bal_0, opt_0], axis=1)
    final_1 = pd.concat([total_bal_1, opt_1], axis=1)

    final = pd.concat([final_0, final_1], axis=0)
    final = pd.concat([final, target_total], axis=0)

    return final.sort_values(['POINT', 'WR_DT'])
