import pandas as pd
import numpy as np


def ENT_FUNCTION(data, INDR, year):  # 입력: HDR데이터, 업종데이터, 연도

    # YEAR에 해당하는 연도 데이터 추출

    df = data[data.WR_DT.apply(lambda x: str(x)[:4]) == year][['SUP_RGNO', 'SUP_NM', 'BYR_RGNO', 'BYR_NM', 'SUP_AMT']]

    # SUP업종 추출

    sup_indr = pd.merge(df, INDR, left_on=['SUP_RGNO'], right_on=['COM_RGNO'], how='left')[
        ['SUP_RGNO', 'SUP_NM', 'BIZ_INDR_CD', 'BYR_RGNO', 'BYR_NM', 'SUP_AMT']]
    sup_indr.columns = ['SUP_RGNO', 'SUP_NM', 'SUP_INDR_CD', 'BYR_RGNO', 'BYR_NM', 'SUP_AMT']

    # BYR업종 추출 및 개인사업자 제거

    sup_byr_indr = pd.merge(sup_indr, INDR, left_on=['BYR_RGNO'], right_on=['COM_RGNO'], how='left')[
        ['SUP_RGNO', 'SUP_NM', 'SUP_INDR_CD', 'BYR_RGNO', 'BYR_NM', 'BIZ_INDR_CD', 'SUP_AMT']]
    sup_byr_indr.columns = ['SUP_RGNO', 'SUP_NM', 'SUP_INDR_CD', 'BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'SUP_AMT']

    sup_byr_indr_ = sup_byr_indr[byr_indr.BYR_RGNO != -100000000000]

    # SUP, BYR 각각 총 거래금액 산출

    sup_sum = sup_byr_indr_.groupby(['SUP_RGNO'])['SUP_AMT'].sum().reset_index(drop=False)
    sup_sum.columns = ['SUP_RGNO', 'SUP_TOT']

    byr_sum = sup_byr_indr_.groupby(['BYR_RGNO'])['SUP_AMT'].sum().reset_index(drop=False) 
    byr_sum.columns = ['BYR_RGNO', 'BYR_TOT']

    # 거래관계데이터에서 각 업체의 총 거래금액 JOIN

    join1 = pd.merge(sup_byr_indr_, sup_sum, how='left')
    join2 = pd.merge(join1, byr_sum, how='left')

    # SUP 업체의 각각의 ENTROPY 산출 (기준 : 거래 수)

    sup_ct1 = join2.groupby(['SUP_RGNO', 'SUP_INDR_CD', 'BYR_INDR_CD']).count().iloc[:, :4].reset_index(drop=False)
    sup_ct1.columns = ['SUP_RGNO', 'SUP_INDR_CD', 'BYR_INDR_CD', 'SUP_CN']

    sup_ct2 = sup_ct1.groupby(['SUP_RGNO', 'SUP_INDR_CD'])['SUP_CN'].sum().reset_index(drop=False)
    sup_ct2.columns = ['SUP_RGNO', 'SUP_INDR_CD', 'SUP_SM']

    sup_ct3 = pd.merge(sup_ct1, sup_ct2, how='left')

    sup_ct3['SUP_P'] = sup_ct3['SUP_CN'] / sup_ct3['SUP_SM']
    sup_ct3['SUP_ENT'] = np.log2(sup_ct3['SUP_P']) * (-1) * sup_ct3['SUP_P']

    sup_ct4 = sup_ct3.groupby(['SUP_RGNO', 'SUP_INDR_CD'])['SUP_ENT'].sum().reset_index(drop=False)
    sup_ct4.columns = ['SUP_RGNO', 'SUP_INDR_CD', 'SUP_ENT_SUM']

    sup_ct5 = pd.merge(sup_ct3, sup_ct4, how='left')

    # 거래데이터에 JOIN

    join3 = pd.merge(join2, sup_ct5, how='left')

    # BYR 업체의 각각의 ENTROPY 산출 (기준 : 거래 수)

    byr_ct1 = join2.groupby(['BYR_RGNO', 'BYR_INDR_CD', 'SUP_INDR_CD']).count().iloc[:, :4].reset_index(drop=False)
    byr_ct1.columns = ['BYR_RGNO', 'BYR_INDR_CD', 'SUP_INDR_CD', 'BYR_CN']

    byr_ct2 = sup_ct1.groupby(['BYR_RGNO', 'BYR_INDR_CD'])['BYR_CN'].sum().reset_index(drop=False)
    byr_ct2.columns = ['BYR_RGNO', 'BYR_INDR_CD', 'BYR_SM']

    byr_ct3 = pd.merge(byr_ct1, byr_ct2, how='left')

    byr_ct3['BYR_P'] = byr_ct3['BYR_CN'] / byr_ct3['BYR_SM']
    byr_ct3['BYR_ENT'] = np.log2(byr_ct3['BYR_P']) * (-1) * byr_ct3['BYR_P']

    byr_ct4 = sup_ct3.groupby(['BYR_RGNO', 'SUP_INDR_CD'])['BYR_ENT'].sum().reset_index(drop=False)
    byr_ct4.columns = ['BYR_RGNO', 'BYR_INDR_CD', 'BYR_ENT_SUM']

    byr_ct5 = pd.merge(byr_ct3, byr_ct4, how='left')

    # 거래데이터에 JOIN

    join4 = pd.merge(join3, byr_ct5, how='left').drop_duplicates()

    # 필요한 컬럼만 추출

    Total = join4[
        ['SUP_RGNO', 'SUP_NM', 'SUP_INDR_CD', 'SUP_TOT', 'SUP_P', 'BYR_RGNO', 'BYR_NM', 'BYR_INDR_CD', 'BYR_TOT',
         'BYR_P', 'SUP_ENT_SUM', 'BYR_ENT_SUM']].drop_duplicates()

    return Total



