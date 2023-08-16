import pandas as pd

def file(q1, q2, q3, q4, col):

    d1 = q1[col]
    d2 = q2[col]
    d3 = q3[col]
    d4 = q4[col]

    d1.columns = ['상가업소번호', '1분기_상호명', '1분기_분류']
    d2.columns = ['상가업소번호', '2분기_상호명', '2분기_분류']
    d3.columns = ['상가업소번호', '3분기_상호명', '3분기_분류']
    d4.columns = ['상가업소번호', '4분기_상호명', '4분기_분류']

    m1 = pd.merge(d1, d2, how='outer')
    m2 = pd.merge(m1, d3, how='outer')
    m3 = pd.merge(m2, d4, how='outer')

    return m3

# 분기별 모든 데이터 , 각 컬럼 이름

def ind(data, n1, n2, n3, n4):
    # 1년동안 같은 업종으로 유지한 데이터 추출
    TF = (data[n1] == data[n2]) & (data[n2] == data[n3]) & (data[n3] == data[n4])

    year_data = data[TF].reset_index(drop=True)
    ch_data = data[data.상가업소번호.isin(data[TF].상가업소번호.tolist()) == False].reset_index(drop=True)

    return year_data, ch_data


# 4분기 소분류명이 있다고 가정. (상가업소번호, 각 분기 소분류명, 1년이상 데이터 유지)
# 값 기준은 1분기~4분기 업종이 한번이라도 바뀐 값으로 가정

def max_indr(data, n1, n2, n3, n4):
    data = data.reset_index(drop=True)

    ch = []  # 바뀐 값
    same = []  # 같은 값

    for i in data.index:

        com_indr = pd.DataFrame(data.loc[i, [n1, n2, n3, n4]])
        com_indr['CN'] = 0
        com_group = com_indr.groupby(com_indr.columns.tolist()[0]).count().sort_values('CN', ascending=False)

        if com_group.count()[0] == 1:
            ch.append(i)
            continue;

        # 유지된 값 중, 분기별 업종의 수가 가장 많은 값으로 변환
        if com_group['CN'].iloc[0] > com_group['CN'].iloc[1]:

            data.loc[i, [n1, n2, n3, n4]] = com_group.index[0]
            ch.append(i)

            # 아닌 경우(2:2인 경우), 다른값으로 입력
        else:

            same.append(i)

    li = data[data.index.isin(ch)].상가업소번호  # 바꾼 상가업소 리스트 추출
    s_data = data[data.index.isin(same)]  # 바뀌지 않은 값 추출

    return data, li, s_data  # 변환데이터, 바뀐데이터상가업소번호, 바뀌지 않은 값 추출


# data1: 위의 함수에서 변환되지 않은 데이터(s_data), data2: 1년동안 유지된 대분류 데이터

def max_indr2(data1, data2, n1, n2, n3, n4):  # 소분류 데이터, 대분류 데이터 , 분기별 소분류명 컬럼이름

    li = data1.상가업소번호.tolist()
    target = data2[data2.상가업소번호.isin(li)]

    ch_data, s_li, s_data = max_indr(target, n1, n2, n3, n4)

    ex_data = data1[data1.상가업소번호.isin(s_li)].reset_index(drop=True)

    n_data = data1[data1.상가업소번호.isin(s_li) == False].reset_index(drop=True)

    # 마지막 분기 기준으로 값 변환

    ch_g = ex_data[n4]
    ex_data[n3] = ch_g
    ex_data[n2] = ch_g
    ex_data[n1] = ch_g

    total = pd.concat([ex_data, n_data], axis=0)

    return total, n_data


# 상호명으로 데이터 바꾸기 <조건 1년이상 업체가 유지된 경우에만 진행> Year_data
# 컬럼 통일해서 진행, 분기별상호명, 분기별 업종명, 상가업소번호 필요 / 4분기가 모두 같은 업종일 경우 제외

def name_ch_data_list(data, D):  # 업소데이터, 1년동안 유지된 대분류 데이터

    # 1년동안 유지된 소분류명 추출
    df = data.iloc[:, :3]
    df.columns = ['상가업소번호', '상호명', '소분류']

    # 상호가 같은데 소분류가 2개이상인 상호명 추출
    group1 = df.groupby(['상호명', '소분류']).count().reset_index(drop=False)
    group2 = group1.groupby('상호명').count()
    group2_name = group2[group2['소분류'] >= 1].index.tolist()
    group3 = group1[group1.상호명.isin(group2_name)]

    # 같은 상호명에서 소분류 업종수가 가장 많은 값으로 추출
    group4 = group3.groupby('상호명')['상가업소번호'].max().reset_index(drop=False)

    # 상호명이 여러개이면서 소분류도 여러개인 값 추출
    group5 = pd.merge(group3, group4)
    group5.columns = ['상호명', '바꿀소분류', '갯수']

    # 위 상호명 거른 것 중, 상호명이 겹치면서 분류가 다른 값들 추출
    g = group5.groupby('상호명').count().reset_index(drop=False)
    name = g[g['갯수'] > 1]['상호명'].tolist()

    # 해당 상호명의 대분류를 비교하여 분류가 같다면 하나의 소분류로 진행
    check_name = D[D['1분기_상호명'].isin(name)]['1분기_상호명'].tolist()
    c1 = group5[group5['상호명'].isin(check_name)].reset_index(drop=True)
    c2 = c1[['상호명']].drop_duplicates().index.tolist()
    C = c1[c1.index.isin(c2)]

    # 위에서 걸러 내었을 때, 대분류가 같지 않은 경우, 값에서 제외
    CN = group5[group5.상호명.isin(C.상호명.tolist()) == False]

    del_name = CN.groupby('상호명')['갯수'].count()
    Del_Name = del_name[del_name > 1].index.tolist()

    N = CN[CN.상호명.isin(Del_Name) == False]

    # 최종적으로 만든 리스트 추출
    T = pd.concat([C, N], axis=0).iloc[:, :2]

    return T

def Name_Ch_data_fin(f_data, Name_data, Year_data):  # 4분기 데이터 중 분류하지 못한데이터, 상호명 정제데이터, 위에서 정제한 데이터

    # 분기별 상호명이 모두 같을 때에만, 위에서 정제한 상호명데이터 정제진행

    tf = (f_data['1분기_상호명'] == f_data['2분기_상호명']) & (f_data['1분기_상호명'] == f_data['3분기_상호명']) & (
                f_data['3분기_상호명'] == f_data['4분기_상호명'])
    f_target = f_data[tf][['상가업소번호', '4분기_상호명']]
    f_target.columns = ['상가업소번호', '상호명']

    # 상호명으로 정제
    name_mer1 = pd.merge(f_target, Name_data)

    # 이름으로 바꾼 값 이외에는, 가장 최근의 값으로 진행
    ff_data = f_data[f_data.상가업소번호.isin(name_mer1.상가업소번호.tolist()) == False].reset_index(drop=True)
    fin_name = ff_data['4분기_분류']
    ff_data['3분기_분류'] = fin_name
    ff_data['2분기_분류'] = fin_name
    ff_data['1분기_분류'] = fin_name

    # 최근데이터도 추가
    group1 = ff_data[['4분기_상호명', '4분기_분류']]
    group1.columns = Name_data.columns
    group2 = group1[group1.상호명.isin(Name_data.상호명.tolist()) == False].drop_duplicates()
    g2_cn = group2.groupby('상호명').count().sort_values('바꿀소분류', ascending=False)
    del_list = g2_cn[g2_cn['바꿀소분류'] > 1].index.tolist()
    group3 = group2[group2.상호명.isin(del_list) == False]
    group = pd.concat([Name_data, group3], axis=0).drop_duplicates()

    # 데이터 고정
    name_mer1.columns = ['상가업소번호', '1분기_상호명', '1분기_분류']

    cn = 0
    for i in ff_data.columns[3:]:
        cn = cn + 1
        if np.mod(cn, 2) == 0:
            name_mer1[i] = name_mer1['1분기_분류']
        else:
            name_mer1[i] = name_mer1['1분기_상호명']

    # 모두 바꾼값으로 진행
    total_mer = pd.concat([name_mer1, ff_data], axis=0)

    action2 = pd.concat([Year_data, total_mer], axis=0)

    return group, action2  # 정제데이터, 1년치 데이터 추출


# 데이터 바꾸기 : 원본데이터, 1년치데이터, 상호명 정제데이터, 소상공인업종데이터

def fin_data(bef_data, Year_total_data, Name_data, indr_list):

    # 컬럼 제거 후, 소분류명에 따른 업종데이터로 업데이트
    del_col = ['상권업종대분류코드', '상권업종대분류명', '상권업종중분류코드', '상권업종중분류명', '상권업종소분류코드', '상권업종소분류명', '표준산업분류코드', '표준산업분류명']
    res_col = bef_data.columns

    # 4분기동안 있던 데이터와, 신규데이터로 나눠서 정제 진행
    c1 = bef_data[bef_data.상가업소번호.isin(Year_total_data.상가업소번호.unique().tolist())]
    c2 = bef_data[bef_data.상가업소번호.isin(Year_total_data.상가업소번호.unique().tolist()) == False]

    # 삭제 후 , 변경된 소상공인소분류명으로 정제 실행
    for i in del_col:
        del c1[i]

    C1 = pd.merge(c1, Year_total_data)
    CC1 = pd.merge(C1, indr_list)
    CC1 = CC1[res_col]

    # 상호명 데이터로 정제할 때, 새로운 데이터에 해당되는 상호명만 정제 실행
    Name_data.columns = ['상호명', '상권업종소분류명']

    ch_c2 = c2[c2.상호명.isin(Name_data.상호명.tolist())]
    els_c2 = c2[c2.상호명.isin(Name_data.상호명.tolist()) == False]

    for i in del_col:
        del ch_c2[i]

    ch_c2_mer = pd.merge(ch_c2, Name_data)
    total_c2 = pd.merge(ch_c2_mer, indr_list)
    total_c2 = total_c2[res_col]

    res = pd.concat([CC1, total_c2, els_c2], axis=0)

    return res