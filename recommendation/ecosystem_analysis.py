import pandas as pd

# 지정한 업종에서 거래 수가 가장 많은 Top1을 뽑아 Value_Chain 형성

def deal_most_vcn(data, start_indr, indr_y, rgno_indr, dic): # 데이터(16~18년도 거래데이터), 리스트 시작업종, 필터업종리스트, 기준업종, 업종설명파일

    sup_rgno = data[data['BYR_INDR_CD'] == rgno_indr]['BYR_RGNO'].unique().tolist()
    indr_list = [start_indr]

    lev = 0
    while 1:
        try:
            # 이전 단계 업종 리스트의 수
            before = len(indr_list)

            # 마지막 단계의 업종이면서 car_rgno에 해당하는 업체들과 거래하는 업체 추출
            sup_data = data[(data['BYR_RGNO'].isin(sup_rgno) & data['SUP_INDR_CD'].str.contains(indr_list[-1]))]
            sup = sup_data['SUP_RGNO'].unique()

            # 추출된 업체들 중 같은 업체와 거래하는 업체값 제거
            de_sup = []
            for i in sup:
                if i not in sup_rgno:
                    de_sup.append(i)

            # 위에서 정제한 업체들과 거래하는 업종 추출 (거래횟수가 가장 많은 값으로 정렬)
            indr = data[data['BYR_RGNO'].isin(de_sup)].groupby('SUP_INDR_CD').count().reset_index(drop=False)
            indr = indr[['SUP_INDR_CD', 'SUP_RGNO']].sort_values('SUP_RGNO', ascending=False)

            # 추출한 업종들 중 Filter 리스트에 포함되면서 indr_list에 없는 업종을 추가
            for i in indr['SUP_INDR_CD']:
                x = i[0:3]
                if x not in indr_y:
                    continue;
                else:
                    if i not in indr_list:
                        indr_list.append(i)
                        break;
                    else:
                        continue;

            # 이후 단계 업종 리스트의 수
            after = len(indr_list)

            # 이전 단계의 수와 이후 단계의 수가 같다면 Filter가 되지 않은 것이므로 종료
            if before == after:
                print(lev)
                break;
            lev = lev + 1
            # 수정된 업체수를 sup_rgno로 전달
            sup_rgno = de_sup


        except:
            print(lev, "error")
            break

    # 각 단계 업종에 해당하는 업종 설명값 추출

    level = pd.DataFrame()
    for i in range(0, len(indr_list)):
        x = pd.DataFrame([indr_list[i], i])
        y = x.T
        level = pd.concat([level, y], axis=0)

    level.columns = ['col', 'LEV']
    result = pd.merge(level, dic, how='left')

    return result

# 해당업체와 거래하는 업체들 중 가장 많은 공급액을 주는 업체를 뽑으면서 전체의 거래망을 보는 알고리즘

def most_sup_p_vcn(SUP_P, target, indr_list, dic): # 사업자, 필터업종리스트, 2018년 거래매출데이터, 업종설명파일
    # 각 업체단계 표시
    lev = 0

    # 가장 공급액이 높은 업체 추출
    result = SUP_P[SUP_P['SUP_RGNO'] == target][['SUP_RGNO', 'SUP_INDR_CD']].drop_duplicates()

    while 1:
        try:
            # 이전 단계 업체의 수
            before = len(result)

            # 마지막 단계에 해당 업체와 거래하는 업체 추출 (거래액이 높은 값으로 정렬)
            final_rgno = [result.SUP_RGNO.iloc[-1]]
            group = SUP_P[SUP_P['BYR_RGNO'].isin(final_rgno)].groupby('SUP_RGNO')['SUP_AMT'].sum().sort_values(ascending=False).reset_index(drop=False)

            # 거래 업체들 중 필터 업종에 해당하면서 result값에 없는 업체 추출
            for i in group['SUP_RGNO']:
                indr = SUP_P[SUP_P['SUP_RGNO'] == i]['SUP_INDR_CD'].iloc[0]
                t = indr[0:3]
                if t in indr_list:
                    if i not in list(result['SUP_RGNO']):
                        tar = pd.DataFrame([i, indr]).T
                        tar.columns = ['SUP_RGNO', 'SUP_INDR_CD']
                        result = pd.concat([result, tar], axis=0)
                        break

            # 이후 단계 업체의 수
            after = len(result)

            # 이전 단계의 수와 이후 단계의 수가 같다면 Filter가 되지 않은 것이므로 종료
            if before == after:
                print(lev)
                break;
            lev = lev + 1
        except:
            print(lev, "error")
            break

    # 각 단계의 레벨 설정 후 업종 설명 값 추출

    result['level'] = [i for i in range(0, len(result))]
    dic.columns = ['SUP_INDR_CD', 'EX']
    total = pd.merge(result, dic).sort_values('level')

    return total

