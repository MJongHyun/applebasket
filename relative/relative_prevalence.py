import pandas as pd
import numpy as np

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
