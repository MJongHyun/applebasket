import pandas as pd
import string
import re


int_key = {0:'영', 1:'일', 2:'이', 3:'삼', 4:'사', 5:'오', 6:'육', 7:'칠', 8:'팔', 9:'구'}

comma = ','.join(string.ascii_letters)
abc_li = comma.split(',')

abc_key = {}
h = ['에이', '비', '씨', '디', '이', '에프', '쥐', '에이치', '아이', '제이', '케이', '엘', '엠', '엔', '오', '피', '큐', '알', '에스', '티', '유',
     '브이', '더블유', '엑스', '와이', '제트']
for i in range(0, len(h)):
    abc_key[abc_li[i]] = h[i]

for i in range(0, len(h)):
    j = i + 26
    abc_key[abc_li[j]] = h[i]

# 숫자,영어 변환

def int_abc(num_name, int_keys = int_key, abc_keys = abc_key):

    S = []

    for name in num_name:

        N = name
        spl = []

        for n in N:

            try:
                if int(n) in int_keys.keys():
                    new_n = int_keys[int(n)]
                    spl.append(new_n)
            except:
                if n in abc_keys.keys():
                    new_al = abc_keys[n]
                    spl.append(new_al)
                else:
                    spl.append(n)

        s_name = ''.join(spl)
        S.append(s_name)

    X_name = pd.DataFrame(S, index=num_name.index)

    return X_name

# 상권 하나당 같은주소에 있는 값들 추려내기

def test_func(com_id, data, int_keys = int_key, abc_keys = abc_key):

    test = data[data.COMMERCIAL_ID == com_id]

    # 한 상권에서 같은 업종으로 가지고 있는 값 추출
    for i in test.지번주소:

        same_addr = test[test.지번주소 == i]

        # 상호명,지점명 에러가 있었기에 지점명도 상호명에 포함시켜 값을 사용, 단, 지점명에 점과 같은 필요없는 값 제거
        same_addr['지점명'] = same_addr['지점명'].fillna('')
        same_addr.loc[same_addr.지점명.str.endswith('점'), '지점명'] = ''

        same_addr['상호'] = same_addr['상호명'] + same_addr['지점명']

        # 상호명에 있는 특수문자,띄어쓰기 값 모두 제거
        for i in same_addr['상호']:
            tar = i
            get = re.sub(pattern='[^\w\s\" "]', repl='', string=tar)
            same_addr.loc[same_addr['상호'] == tar, '상호'] = get

        # 상호명에 숫자나 영어가 있는 경우 수정을 위해 추출
        tf = (same_addr['상호'].str.contains('|'.join(string.digits))) | (
            same_addr['상호'].str.contains('|'.join(string.ascii_letters)))
        num_name = same_addr[tf]['상호']

        X_name = int_abc(num_name, int_keys, abc_keys)

        for i in X_name.index:
            same_addr.loc[same_addr.index == i, '상호'] = X_name.loc[i, 0]

        break

    return same_addr


from soynlp.hangle import jamo_levenshtein


def lev_fuc(same_addr):
    res = pd.DataFrame()

    # 상호명을 기준으로 편집거리 알고리즘 실행
    for i in same_addr['상호']:

        start = same_addr[same_addr['상호'] == i]['상호'].iloc[0]
        els = same_addr[same_addr['상호'] != i]

        for j in els['상호']:
            end = j
            p = jamo_levenshtein(start, end)
            r = pd.DataFrame([start, end, p]).T
            res = pd.concat([res, r], axis=0)

    res.columns = ['상호1', '상호2', '값']

    res_min = res.groupby('상호1')['값'].min().reset_index(drop=False)

    R = pd.merge(res, res_min)

    return res, R


def simi_fuc(same_addr):
    
    same_addr['층정보'] = same_addr['층정보'].fillna('')

    res1 = pd.DataFrame()

    for i in same_addr['상호']:
        els = same_addr[same_addr['비교값'] != i]
        bi = els[els['상호'].str.contains(i)][['상호']]
        bi['비교값'] = i

    res1 = pd.concat([res1, bi], axis=0)

    res2 = pd.DataFrame()

    for i, j in res1.values:

        n1 = same_addr[same_addr['상호'] == i]['층정보'].iloc[0]
        n2 = same_addr[same_addr['상호'] == j]['층정보'].iloc[0]

        if n1 == n2:
            nn = pd.DataFrame([i, j]).T
            res2 = pd.concat([res2, nn], axis=0)

    return res2