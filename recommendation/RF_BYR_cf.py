import numpy as np
import pandas as pd


def rgno_cor(X, Y, tf):  # 입력값 : 검색하는 회사, 다른 회사, 두 회사가 같은 업종에 거래하고 있는지 아닌지 파악여부

    # 두 업체가 거래하는 같은 업종의 수
    N = len(tf[tf == True])

    # N = 1 인 경우 , 상관관계 식에 의해 분자가 0이 되므로 0으로 추출
    if N == 1:
        return 0

    sumX = np.sum(X)
    sumY = np.sum(Y)

    sumXsumY = sumX * sumY
    sumXY = np.sum(np.array(X) * np.array(Y))

    sumX2 = np.sum((X * X))
    sumY2 = np.sum((Y * Y))

    pX = pow(sumX, 2)
    pY = pow(sumY, 2)

    # 분자, 분모 추출 후 에러 있을 시, 0으로 추출
    A = sumXY - (sumXsumY / N)
    B = np.sqrt((sumX2 - (pX / N)) * (sumY2 - (pY / N)))

    if A == 0 or B == 0:
        return 0

    else:
        his = A / B

        return his


def RF_BYR_cf(rgno, Data):  # 입력값 : 회사 사업자, 거래데이터

    Data = Data[Data['P'] != 1]

    # 입력회사와 같은 업종과 거래하는 회사의 사업자 번호 추출
    indr = Data[Data['SUP_RGNO'] == rgno]['SUP_INDR_CD'].unique().tolist()
    byr_com = Data[Data['SUP_RGNO'] == rgno]['BYR_RGNO'].unique().tolist()
    byr_indr = Data[Data['SUP_RGNO'] == rgno].groupby('BYR_INDR_CD').count().index.tolist()

    # 입력회사와 같은 업종인 회사의 업체들을 추출한 후, 각 업체들이 거래하고 있는 회사의 업종과 매츌액을 추출
    target = Data[Data['SUP_INDR_CD'].str.contains('|'.join(indr))][['SUP_RGNO', 'BYR_INDR_CD', 'BYR_TOT']]
    tar = target[target['BYR_INDR_CD'].str.contains('|'.join(byr_indr))]
    T = tar.pivot_table(values='BYR_TOT', index=['SUP_RGNO'], columns=['BYR_INDR_CD'], aggfunc='sum').fillna(0)

    # 입력회사와 다른 회사의 상관관계를 알기위해 데이터 추출
    x_r = T[T.index == rgno]
    y_r = T[T.index != rgno]

    # 입력회사와 같은 업종과 거래하는 것을 기준으로 상관관계 값 추출
    cor = {}
    for i in y_r.index:

        t = y_r[y_r.index == i]
        tf = (np.array(t) != 0) & (np.array(x_r) != 0)

        if t[tf].count()[0] == 0:
            cor[i] = 0

        else:
            X = np.array(x_r)[tf]
            Y = np.array(t)[tf]
            num = rgno_cor(X, Y, tf)
            cor[i] = num

    # 상관관계 추출한 데이터 들 중 0.6 이상 (강한 상관관계)를 가지는 값들을 추출
    cor_data = pd.DataFrame(cor.items(), columns=['SUP_RGNO', 'COR'])
    cor_hi = cor_data[cor_data['COR'] >= 0.6]['SUP_RGNO']

    # 상관관계가 높은 업체들 리스트 추출
    n_sup = Data[Data.SUP_RGNO.isin(cor_hi.tolist())]
    sup_info = n_sup[['SUP_RGNO', 'SUP_NM', 'SUP_INDR_CD']]

    # 높은 업체들 중 검색회사와 거래하지 않는 업체의 데이터 추출
    n_byr = n_sup[n_sup.BYR_RGNO.isin(byr_com) == False]

    # 검색회사와 같은 업종의 거래를 하는 값만 추출
    total = n_byr[n_byr.BYR_INDR_CD.str.contains('|'.join(byr_indr))][
        ['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P']]

    return total.sort_values('BYR_TOT', ascending=False).head(20)