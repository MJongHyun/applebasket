import pandas as pd
import numpy as np


# 같은 업종이면서 거래하지 않는 회사의 리스트를 구매력 순서로 추출

def RF_BYR_ver_1(RGNO, COM): # 입력데이터 : 타겟회사의 사업자, 전체거래 데이터

    data = COM
    data = data[data['P'] != 1]  # 1인 값 제거
    rg = RGNO
    cd = data[data['SUP_RGNO'] == rg]['SUP_INDR_CD'].iloc[0]  # 검색하는 회사의 표준분류코드

    # 검색하는 회사와 표준산업분류코드가 같고, 검색하는 회사를 제외한 SUP_LIST로 추출
    sup_list = data[data['SUP_INDR_CD'].isin([cd])]
    sup_list = sup_list[sup_list['SUP_RGNO'] != rg]
    sup_list = sup_list['SUP_RGNO'].unique()

    rgno_byr = data[data['SUP_RGNO'] == rg]['BYR_RGNO']  # 검색하는 회사의 BYR 추출
    sup_list_data = data[data['SUP_RGNO'].isin(list(sup_list))]  # SUP_LIST에 있는 회사와 거래하는 모든 값 추출

    # 검색하는 회사와 거래하지 않는 BYR 리스트 추출
    byr_list = sup_list_data[sup_list_data['BYR_RGNO'].isin(list(rgno_byr)) == False]

    # byr_list 들 중 검색하는 회사와 같은 업종의 거래를 하는 값들 추출
    rgno_byr_cd = data[data['SUP_RGNO'] == rg]['BYR_INDR_CD']
    new_byr_list = byr_list[byr_list['BYR_INDR_CD'].isin(list(rgno_byr_cd))][
        ['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P']]

    new_byr_list = new_byr_list[new_byr_list['BYR_RGNO'] != rg]  # 거래 회사목록에 자기자신 제거

    # new_byr_list 를 group 화하여 총 BYR_TOT합, CN 추출
    group_sum = new_byr_list.groupby(['BYR_RGNO', 'BYR_INDR_CD'])['BYR_TOT', 'P'].sum()
    group_cn = new_byr_list.groupby(['BYR_RGNO', 'BYR_INDR_CD'])['BYR_NM'].count()
    group_li = pd.concat([group_sum, group_cn], axis=1)
    group_li = group_li.reset_index(drop=False)
    group_li.columns = ['BYR_RGNO', 'BYR_INDR_CD', 'BYR_TOT', 'P', 'CN']

    # 기업명을 추출하기 위해 merge 후 unique한 값만 추출
    group_mer = pd.merge(group_li, new_byr_list[['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD']])
    total_mer = group_mer.iloc[list(group_mer[['BYR_RGNO', 'BYR_INDR_CD']].drop_duplicates().index), :][
        ['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P', 'CN']]

    return total_mer.sort_values('BYR_TOT', ascending=False).iloc[:20, :]  # BYR_TOT 기준으로 상위 20개 추출

# 주력 업종이 비슷하면서 거래하지 않는 회사의 리스트 추출

def RF_BYR_2_ver_1(RGNO, COM): # 입력데이터 : 타겟회사의 사업자, 전체거래 데이터

    data = COM
    data = data[data['P'] != 1]  # 1인 값 제거
    rg = RGNO
    cd = data[data['SUP_RGNO'] == rg]['SUP_INDR_CD'].iloc[0]  # 검색하는 회사의 표준분류코드

    # 검색하는 회사의 주력사업 표준산업분류코드
    sup_rgno_most_byr_cd = data[data['SUP_RGNO'] == rg].sort_values('P', ascending=False)['BYR_INDR_CD'].iloc[0]
    rgno_byr = data[data['SUP_RGNO'] == rg]['BYR_RGNO']  # 검색하는 회사가 거래하는 BYR의 사업자 번호 추출

    byr_list = data[data['BYR_INDR_CD'] == sup_rgno_most_byr_cd]  # 주력 표준산업분류코드로 거래하는 회사들 추출
    new_byr_list = byr_list[byr_list['BYR_RGNO'].isin(list(rgno_byr)) == False]  # 회사들 중 검색하는 회사와 거래하지 않는 회사들 추출

    # 위에서 나온 new_byr_list에서 검색하는 회사와 같은 산업분류코드를 가진 값들의 리스트 추출
    new_byr_list = new_byr_list[new_byr_list['SUP_INDR_CD'].isin([cd])][
        ['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P']]
    new_byr_list = new_byr_list[new_byr_list['BYR_RGNO'] != rg]  # 거래 회사목록에 자기자신 제거

    # new_byr_list 를 group 화하여 총 BYR_TOT합, CN 추출
    group_sum = new_byr_list.groupby(['BYR_RGNO', 'BYR_INDR_CD'])['BYR_TOT', 'P'].sum()
    group_cn = new_byr_list.groupby(['BYR_RGNO', 'BYR_INDR_CD'])['BYR_NM'].count()
    group_li = pd.concat([group_sum, group_cn], axis=1)
    group_li = group_li.reset_index(drop=False)
    group_li.columns = ['BYR_RGNO', 'BYR_INDR_CD', 'BYR_TOT', 'P', 'CN']

    # 기업명을 추출하기 위해 merge 후 unique한 값만 추출
    group_mer = pd.merge(group_li, new_byr_list[['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD']])
    total_mer = group_mer.iloc[list(group_mer[['BYR_RGNO', 'BYR_INDR_CD']].drop_duplicates().index), :][
        ['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P', 'CN']]

    return total_mer.sort_values('BYR_TOT', ascending=False).iloc[:20, :]  # BYR_TOT 기준으로 상위 20개 추출


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


def rgno_cf_ver0(rgno, Data):  # 입력값 : 회사 사업자, 거래데이터

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

    # 아모레퍼시픽과 같은 업종의 거래를 하는 값만 추출
    total = n_byr[n_byr.BYR_INDR_CD.str.contains('|'.join(byr_indr))][['BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT', 'P']]

    return total.sort_values('BYR_TOT', ascending=False).head(20) # BYR_TOT 기준으로 상위 20개 추출


