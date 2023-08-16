"""
EW_CD 데이터에서, 위험1, 위험2, 위험3, 파산을 등급들 중, 월에 가장 높은 등급을 받은 것을 기준으로 정제하여
시간에 따른 업체의 등급 변화 (주기: 3달)를 통해 나온 결과 값을 바탕으로 진행
"""

import pandas as pd
import numpy as np
import collections

# 데이터 정제 : 업체별 월별 EW_CD가 가장 높은 데이터를 추출
# 데이터 컬럼 : 월별, 사업자등록번호, EW_CD, EW_CD 등록 날짜

def refine_data(data): # 입력값 : EW_CD 관련 데이터

    # 데이터에서 등급 값이 위험1, 위험2, 위험3, 파산인 (순서대로 2,3,4,6) 데이터만 추출
    df = data[data.EW_CD.isin([2, 3, 4, 6])]

    # 데이터에서 월별 업체가 가장 높은 값을 받은 등급만을 가져오기 위해 정렬

    sort_data = df[['M_DT', 'COM_RGNO', 'EW_CD']].sort_values(['COM_RGNO', 'M_DT', 'EW_CD'], ascending=False).reset_index(
        drop=True)
    ind = sort_data[['M_DT', 'COM_RGNO']].drop_duplicates().index

    Data = df[sor_data.index.isin(ind)].sort_values(['COM_RGNO','M_DT'])

    # 정렬한 데이터를 pivot table로 만들고 추출

    Data_pivot = Data.pivot('COM_RGNO', 'M_DT').iloc[:, 1:]
    com_id = Data_pivot.index

    df_col = []

    for a, y in Data_pivot.columns:
        df_col.append(y)

    total = pd.DataFrame(np.array(Data_pivot), columns=df_col, index=com_id)

    return total, df_col

# 기간데이터이기 때문에 값이 비어있을 경우, 이전에 있던 값으로 채워 값을 진행

def month_check(data, df_col):  # 입력값 : 위 함수를 통해 정제된 pivot table 데이터, 월데이

    res = {}
    DT = {}

    # 업체에 따라 검사한 기간이 다르기 때문에, 업체 검사 시작날짜와 끝 날짜를 추출
    for com in data.index:

        target = data[data.index == com].T

        if len(target.dropna()) == 0:
            continue

        start = target.dropna().index[0]
        end = target.dropna().index[-1]

        az = (start, end)
        DT[com] = az

        a = [i for i, x in enumerate(df_col) if x == start][0]
        z = [i for i, x in enumerate(df_col) if x == end][0]

        # 시작날짜와 끝 날짜 사이에서, nan값이 있는 경우, 그 전달에 나온 EW_CD 적용해서 바꿈
        for i in range(a, z):

            x_ind = target.iloc[i, :].name
            y = target.iloc[i, :].iloc[0]

            if np.isnan(y) == False:

                ch_ = y
                continue;

            else:
                target.loc[target.index == x_ind, com] = ch_

        arr = np.array(target.dropna().T)
        res[com] = arr

    dt = pd.DataFrame(DT.values(), columns=['Q1', 'Q4'])

    result = dt.reset_index(drop=False)

    return res, dt, result

# 한가지 패턴으로만 결과가 나오는 경우 (예) 2,2,2 -> 2 인 경우) 제거

def check_one(res): # 입력 값 : 위에서 추출한 dict (월별 EW_CD 데이터)

    r = res.copy()

    # collections.Counter를 통해, 값이 한가지만 있는 경우 dict에서 제거

    for key, value in res.items():

        li = value[0]

        l = collections.Counter(li)

        if len(l) == 1:
            r.pop(key, None)

    return r

# 나머지 pattern이 있는 데이터의 경우, 월별 데이터 3개에 따른 결과를 만들기 위해, 데이터를 4분기 값으로 추출하여 저장

def q4_pattern_data(r): # 입력값 : check_one함수로 정제한 dict

    # 4분기 값이 되지 않는 업체의 경우, 가장 최근값을 추가하여 4분기 데이터로 만듬
    for com, value in r.items():

        val = value[0]

        if len(val) < 4:

            ep = val[-1]

            if len(val) == 2:

                val = np.append(val, ep)
                val = np.append(val, ep)
                new = np.array([val])

                r[com] = new

            else:

                val = np.append(val, ep)
                new = np.array([val])

                r[com] = new

    # 위에서 만든 데이터를 업체당 4분기 데이터로 만들어 저장

    for com, value in r.items():

        v = value[0]

        if len(v) == 4:

            continue

        else:

            s = 0
            x = np.array([0, 0, 0, 0])

            for i in range(0, len(v) - 4):
                v1 = np.array(v[s:s + 4])
                x = np.vstack((x, v1))
                s = s + 1

        r[com] = x

    return r

# DataFrame으로 추출

def fin_data(r): # 업체당 4분기 등급변화 데이터

    RE = np.array([0, 0, 0, 0, 0])

    for key, value in r.items():

        if len(value) == 1:

            val = np.append(key, value)
            RE = np.vstack((RE, val))

        else:

            for y in value:
                val = np.append(key, y)
                RE = np.vstack((RE, val))

    # 데이터에서 , 4분기의 값이 모두 같은 등급은 경우 제거
    data = pd.DataFrame(RE)
    data.columns = ['COM_RGNO', 'X1', 'X2', 'X3', 'Y']
    del_com = data[(data.X1 == data.X2) & (data.X2 == data.X3) & (data.X3 == data.Y)].index

    Total = data[data.index.isin(del_com) == False].reset_index(drop = True)

    return Total


# 기간데이터 추가

def q1q4_data(Total, result, df_col): # 입력데이터 : 4분기 데이터 추출한 데이터, 기간관련 데이터, 전체기간 컬럼

    # 4분기 추출데이터에 기간을 붙임
    Total['COM_RGNO'] = Total['COM_RGNO'].astype(int)
    result['COM_RGNO'] = result['COM_RGNO'].astype(int)

    T = pd.merge(Total,result)

    # 위에 join한 데이터row 값의 각각 시작점과 끝점을 추
    qq = {}

    for com in T.COM_RGNO.unique():

        I = T[T.COM_RGNO == com]

        qq_li = []

        Q1 = df_col[df_col.index(I.Q1.iloc[0])]

        if Q1 >= 201908:
            Q4 = Q1 + 3
            q1q4 = (Q1, Q4)
            qq_li.append(q1q4)

            qq[com] = qq_li

            continue

        Q4 = df_col[df_col.index(Q1) + 3]

        q1q4 = (Q1, Q4)

        qq_li.append(q1q4)

        for l in range(1, len(I)):

            Q1 = df_col[df_col.index(Q1) + 1]

            if Q1 >= 201908:
                break

            Q4 = df_col[df_col.index(Q1) + 3]

            q1q4 = (Q1, Q4)

            qq_li.append(q1q4)

        qq[com] = qq_li

    # 추출한 데이터와 join한 데이터를 합침

    qq_df = pd.DataFrame()

    for com_, date_ in qq.items():
        q_d = pd.DataFrame(date_)
        q_d['COM_RGNO'] = int(com_)

        qq_df = pd.concat([qq_df, q_d], axis=0)

    qq_df.columns = ['q1', 'q4', 'COM_RGNO']

    ToT = pd.concat([Total, qq_df.reset_index(drop=True)], axis=1).iloc[:, :-1]

    return ToT

"""
회사와의 2차 거래관계를 통해 주거래 관계들의 위험등급에 따라 위험등급이 올라갈수 있다라고 가정
"""


# ew_cd의 해당연도 데이터 추출

import numpy as np
import pandas as pd
import collections


def month_check(data, df_col):  # 입력값 : ew_cd 데이터, 날짜데이터

    res = {}
    DT = {}

    # 업체에 따라 검사한 기간이 다르기 때문에, 업체 검사 시작날짜와 끝 날짜를 추출
    for com in data.index:

        target = data[data.index == com].T

        if len(target.dropna()) == 0:
            continue

        start = target.dropna().index[0]
        end = target.dropna().index[-1]

        az = (start, end)
        DT[com] = az

        a = [i for i, x in enumerate(df_col) if x == start][0]
        z = [i for i, x in enumerate(df_col) if x == end][0]

        # 시작날짜와 끝 날짜 사이에서, nan값이 있는 경우, 그 전달에 나온 EW_CD 적용해서 바꿈
        for i in range(a, z):

            x_ind = target.iloc[i, :].name
            y = target.iloc[i, :].iloc[0]

            if np.isnan(y) == False:

                ch_ = y
                continue;

            else:
                target.loc[target.index == x_ind, com] = ch_

        arr = np.array(target.dropna().T)

        # 날짜에 맞춰 채운 값들을 결과로 추출

        y = []

        for j in range(0, len(arr[0])):
            x = (df_col[a + j], arr[0][j])
            y.append(x)

        res[com] = y

    return res


def EW_CD_DATA(ew_cd, year):  # 입력값 : ew_cd 데이터, 해당연도

    # 위험등급 관련 데이터만 추출

    ew = ew_cd[ew_cd.EW_CD.isin([2, 3, 4, 6])]
    sor = ew.sort_values(['COM_RGNO', 'REG_DT', 'EW_CD'], ascending=False)

    # 가장 최근에 등급을 데이터를 받은 것을 기준으로, 월별 데이터로 추출

    ind = sor[['COM_RGNO', 'M_DT']].drop_duplicates().index
    df = sor[sor.index.isin(ind)][['M_DT', 'COM_RGNO', 'EW_CD']].reset_index(drop=True)

    df['K'] = df.M_DT.astype('str')
    df['N'] = df.K.str.slice(0, 4)

    # 해당연도에 해당되는 값을 추출 후 , month_check 함수를 통해, 검사되지 않았던 달의 경우, 이전에 받았던 등급으로 채움

    target = df[df['N'] == str(year)][['M_DT', 'COM_RGNO', 'EW_CD']]

    T = target.pivot('COM_RGNO', 'M_DT')
    df_col = []

    for i, j in T.columns:
        df_col.append(j)

    t = pd.DataFrame(np.array(T), columns=df_col, index=T.index)

    res = month_check(t, df_col)

    return res


# 위에서 만든 데이터를 dataframe형태로 추출

def fin_data(res):
    data = pd.DataFrame()

    for com, c in res.items():
        r = pd.DataFrame(c, columns=['M_DT', 'EW_CD'])
        r['COM_RGNO'] = com

        data = pd.concat([data, r], axis=0)

    return data


# 거래데이터와 EW_CD 데이터를 통해 회사의 주거래회사들의 등급에 따라 전이위험을 따짐

def RISK_TEST(data, target, rgno, m):  # 입력데이터: 거래데이터, 정제한 ew_cd데이터, 자기사업자, 해당월

    # 현재 자회사의 EW_CD등급 추출

    end_li = []

    m_cd = target[(target.COM_RGNO == rgno) & (target.M_DT == m)]['EW_CD']

    if len(m_cd) == 0:

        m_cd = 0

    else:

        m_cd = m_cd.iloc[0]

    end_li.append(m_cd)

    # 자회사와 가장 많이 거래하는 top10을 추출

    first_rel = data[(data['SUP_RGNO'] == rgno) & (data['M_DT'] == m)][['BYR_RGNO', 'TOT']].sort_values('TOT', ascending=False).head(10)
    fir_li = first_rel.BYR_RGNO.tolist()

    # 위에서 뽑은 10곳의 EW_CD등급을 추출 후, 다시 그 top10의 주거래 top10을 뽑아 EW_CD를 추출
    res = {}

    for i in fir_li:

        test_li = []

        rg = i

        m_cd = target[(target.COM_RGNO == i) & (target.M_DT == m)]['EW_CD']

        if len(m_cd) == 0:

            m_cd = 0

        else:

            m_cd = m_cd.iloc[0]

        test_li.append(m_cd)

        second_rel = data[(data['SUP_RGNO'] == rgno) & (data['M_DT'] == m)][['BYR_RGNO', 'TOT']].sort_values('TOT', ascending=False).head(10)

        if len(second_rel) > 0:

            mer = pd.merge(target, second_rel, left_on='COM_RGNO', right_on='BYR_RGNO')

            if len(mer) > 0:

                ge = mer[mer['M_DT'] == m]['EW_CD'].tolist()

                if len(ge) > 0:
                    test_li.extend(ge)
                    res[rg] = test_li

                else:

                    test_li.append(0)
                    res[rg] = test_li
                    continue
            else:

                test_li.append(0)
                res[rg] = test_li
                continue

        else:

            test_li.append(0)
            res[rg] = test_li
            continue

    # EW_CD을 추출한 값들을 바탕으로 현재자기자신등급 * 0.5 + 각 회사의등급의 평균 * 0.5를 곱하여 결과값 추출
    for r in res.values():
        P = r[0] * 0.5 + np.mean(r[1:]) * (0.5)

        end_li.append(P)

    EW_CD_RES = end_li[0] * 0.5 + np.mean(end_li[1:]) * (0.5)

    return EW_CD_RES


# t-3 ~ t-1 시점에 해당회사와 거래하는 1차,2차 업체들의 등급 추출

def trend_test(data, EW, rgno, DT, li):  # 입력값 : 거래데이터, 월별등급데이터, 사업자, 해당날짜, 기간리스트

    # rgno와 1차거래하는 리스트 추출
    fir_li = data[data['SUP_RGNO'] == rgno]['BYR_RGNO'].tolist()
    fir_ew_cd = EW[EW.COM_RGNO.isin(fir_li)].EW_CD.tolist()

    # 2차거래 리스트 추출
    second_li = data[data['SUP_RGNO'].isin(fir_li)]['BYR_RGNO'].unique().tolist()

    # 1차거래, 2차거래 리스트 추출

    T1 = pd.DataFrame()
    T2 = pd.DataFrame()

    # t-3 ~ t-1시점의 1차거래의 등급 값과 2차거래의 등급 값 추출

    for i in range(3, 0, -1):
        n = li[[i for i, x in enumerate(li) if x == DT][0] - i]

        fst = EW[(EW.COM_RGNO.isin(fir_li)) & (EW.M_DT == n)]['EW_CD'].tolist()
        fst.sort()

        sd = EW[(EW.COM_RGNO.isin(second_li)) & (EW.M_DT == n)]['EW_CD'].tolist()
        sd.sort()

        fst_cn = collections.Counter(fst)
        sd_cn = collections.Counter(sd)

        t1 = pd.DataFrame(fst_cn.values(), index=fst_cn.keys()).reset_index(drop=False)
        t1['M_DT'] = n

        t2 = pd.DataFrame(sd_cn.values(), index=sd_cn.keys()).reset_index(drop=False)
        t2['M_DT'] = n

        T1 = pd.concat([T1, t1], axis=0)
        T2 = pd.concat([T2, t2], axis=0)

    return T1, T2


# 위에서 추출한 데이터를 기반으로, 등급에 따라 가중치 추출

def trend_result(data): # 입력값 : 위에서 추출한 월별 등급데이터

    data = data.reset_index(drop=True)

    # 등급별 (3등급 - 0.2, 4등급 - 0.3, 5,6,7등급 - 0.5) 가중치를 주어, 월별 결과값 추출 (나머지 등급 0으로 처리)

    data['RES'] = 0
    data.loc[data['EW_CD'] == 3, 'RES'] = 0.2
    data.loc[data['EW_CD'] == 4, 'RES'] = 0.3
    data.loc[data['EW_CD'].isin([5, 6, 7]), 'RES'] = 0.5

    data['R'] = data['RES'] * data['EW_CD'] * data['CN']

    T = data.groupby('M_DT')['R'].sum().reset_index(drop=False)

    return T
