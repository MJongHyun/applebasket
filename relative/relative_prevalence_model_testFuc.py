import pandas as pd
import numpy as np


def flavor_network(target):  # 소상공인 중분류추출 데이터

    # 상권에 해당하는 업종 갯수의 합
    target['sum'] = target.sum(axis=1)

    # 상권에서 각 업종이 해당하는 비율 추출

    P = pd.DataFrame()

    for i in range(0, target['sum'].count()):
        x = target.iloc[i, :-1] / target['sum'].iloc[i]
        P = pd.concat([P, x], axis=1)

    P = P.T.reset_index(drop = True)

    # 해당상권의 업종 비율 값과 나머지 상권의 업종 비율의 평균 값의 차 추출

    df = pd.DataFrame()

    for i in range(0, len(P.columns)):  # 각 업종 종류 추출
        num = i  # 각 해당 컬럼의 번호
        D = pd.DataFrame()
        for j in range(0, P.iloc[:, 0].count()):  # 해당 상권 값 추출
            idx = P.index[P.index != P.index[j]]  # 나머지 상권들 추출
            de = pd.DataFrame([P.iloc[P.index[j], num] - P.iloc[idx, num].mean()])
            D = pd.concat([D, de], axis=0)
        df = pd.concat([df, D], axis=1)

    df.columns = P.columns
    df = df.reset_index(drop=True)

    return df


def flavor_network_1(target):  # 소상공인 중분류추출 데이터

    # 상권에 해당하는 업종 갯수의 합
    target['sum'] = target.sum(axis=1)

    # 상권에서 각 업종이 해당하는 비율 추출

    P = pd.DataFrame()

    for i in range(0, target['sum'].count()):
        x = target.iloc[i, :-1] / target['sum'].iloc[i]
        P = pd.concat([P, x], axis=1)

    P = P.T.reset_index(drop = True)

    # 해당상권의 업종 비율 값과 나머지 상권의 업종 비율의 평균 값의 차 추출

    df = pd.DataFrame()

    for i in range(0, len(P.columns)):  # 각 업종 종류 추출
        num = i  # 각 해당 컬럼의 번호
        D = pd.DataFrame()
        for j in range(0, P.iloc[:, 0].count()):  # 해당 상권 값 추출
            idx = P.index[P.index != P.index[j]]  # 나머지 상권들 추출
            t = pd.Series(P.iloc[idx, i])  # Series화 한 후 0이 아닌 값의 평균 추출
            non_t = t.iloc[t.nonzero()].mean()
            de = pd.DataFrame([P.iloc[P.index[j], num] - non_t])
            D = pd.concat([D, de], axis=0)
        df = pd.concat([df, D], axis=1)

    df.columns = P.columns
    df = df.reset_index(drop=True)

    return df


def sort_val(t, com_id, i):

    x = t[t.COMMERCIAL_ID == com_id].T
    y = x.iloc[i:, :]
    col = y.columns[0]
    sort = y.sort_values(col, ascending=False)
    re = pd.concat([x.iloc[:i, :], sort])

    return re


def reverse(target):
    tar = target.T
    tar['sum'] = tar.sum(axis=1)

    P = pd.DataFrame()

    for i in range(0, tar['sum'].count()):
        x = tar.iloc[i, :-1] / tar['sum'].iloc[i]
        P = pd.concat([P, x], axis=1)

    P = P.T

    df = pd.DataFrame()

    for i in P.columns:
        num = i
        D = pd.DataFrame()

        for j in P.index:
            idx = P.index[P.index != j]

            de = pd.DataFrame([P.loc[P.index == j, num] - P.iloc[P.index.str.contains('|'.join(idx)), num].mean()])
            D = pd.concat([D, de], axis=1)

        df = pd.concat([df, D], axis=0)

    return df


def rel_pre_ver1(target,com_cn):  # 입력데이터 : 소상공인 중분류 추출 데이터, 상권 총 갯수

    # 상권에서 각 업종이 해당하는 비율 추출
    com_per = np.array(target) / np.array(target.sum(axis=1)).reshape(com_cn, 1)

    # 첫번째 상권의 업종 비율 값과 나머지 상권의 업종비율의 평균 값의 차 추출
    scope = [i for i in range(0, len(com_per))] # 범위

    # 첫번째 값 뽑고 이외의 값 제거
    rank = scope[0]
    del scope[rank]

    # 비율 추출
    rel_pre = com_per[rank] - com_per[scope].mean(axis=0)

    # 다음 비율 값들 추출
    next_scope = [i for i in range(0, len(com_per))]

    for s in range(1, len(com_per)):
        length = next_scope.copy()
        stage = s
        del length[stage]
        next_rel_pre = com_per[stage] - com_per[length].mean(axis=0)
        rel_pre = np.vstack((rel_pre, next_rel_pre))

    data = pd.DataFrame(rel_pre)
    data.columns = target.columns

    return data

"""

def rel_pre(target):  # 입력데이터 : 소상공인 중분류 추출 데이터

    s = pd.DataFrame(np.array(target.sum(axis=1)))

    zero = s[s[0]==0]

    if zero.count()[0]!=0:

        s.loc[s[0]==0,0] = 1
        com_per = np.array(target) / np.array(s)

    else:

        # 상권에서 각 업종이 해당하는 비율 추출
        com_per = np.array(target) / np.array(s)

    # 첫번째 상권의 업종 비율 값과 나머지 상권의 업종비율의 평균 값의 차 추출
    scope = [i for i in range(0, len(com_per))] # 범위

    # 첫번째 값 뽑고 이외의 값 제거
    rank = scope[0]
    del scope[rank]

    # 비율 추출
    rel_pre = com_per[rank] - com_per[scope].mean(axis=0)

    # 다음 비율 값들 추출
    next_scope = [i for i in range(0, len(com_per))]

    for s in range(1, len(com_per)):
        length = next_scope.copy()
        stage = s
        del length[stage]
        next_rel_pre = com_per[stage] - com_per[length].mean(axis=0)
        rel_pre = np.vstack((rel_pre, next_rel_pre))

    data = pd.DataFrame(rel_pre)
    data.columns = target.columns

    return data

"""