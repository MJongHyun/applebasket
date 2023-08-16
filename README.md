# APPLEBASKET

### 목적 : 여러 데이터를 통해 만든 지수를 clustering을 통해 상권을 정의 한 후 상권에 존재하는 업종/업체들의 현상황분석 및 거래 관계를 통한 예측, 추천 분석 등을 보여주는 플랫폼 

### 요약
    
    + applebasketIndex : applebasket 지수 관련 코드  
        - applebasketFuc.py : applebasket 지수 관련 함수 모음
    
    + bankuptcy : 부도예측 관련 코드 
        - bankrupcy_test.py : 부도예측 관련 함수를 만들때 test한 함수들
        
    + clustering : 만든 지수를 통해 상권을 분류하기 위해 사용한 clustering 관련 함수 모음
        - clusteringBaseFuc.py : clustering 관련 test 함수 모음
        - clusteringFuc.py : 실제로 사용했던 함수들 모음
        
    + entropy : 정보량의 기댓값을 나타내는 entropy 이론을 바탕으로 만든 함수들
        - E.N.T.py : entropy를 적용한 것으로 거래데이터에서 주 거래 데이터를 추출하기 위해 만든 함수
        
    + graph : clustering이나 EDA에 사용한 그래프 관련 함수들 모음
        - hist_3d_graph.py : EDA에 사용한 함수들 모음
        - radat_graph.py : clustering한 결과를 radarChart로 보여주기위해 만든 함수
        
     +  linearRegression : applebasket에 적용하진 않았으나 사용한 상관분석과 다르게 보이는 것이 있을까 하여 진행했었음
        - LinearReg_EDA.py : 선형회귀 EDA 관련 함수 모음
        - LinearReg_lasso_model.py : Linear lasso Model 관련 함수
        - LinearReg_refine.py : 선형회귀 함수를 사용하기 위해 정제 함수 모음
      
     + recommendation : 상권에서 거래 데이터 기반으로 업종/업체를 추천하기 위한 코드 모음
        - ecosystem_analysis.py : 거래 기반으로 같은 업종을 바탕으로 업체를 추천하는 함수 
        - MST.py : 최소신장트리 함수, 상권별 특정 업종을 현황을 보여주기 위해 사용한 함수 
        - RF_BYR_cf.py : 상관관계를 이용하여 업체를 추천하는 함수
        - RStest.py : 거래 기반으로 같은 거래를 하는 경쟁업체를 바탕으로 업체를 추천하는 함수
        
     + refine : 사용한 데이터 정제에 사용한 코드 모음
        - company_refine_test.py : 정제 코드 테스트한 함수 모음
        - new_small_data.py : 상호명 정제를 하기위해 사용한 함수
     
     + relative : 상대비교를 통해 추천하는 코드 모음
        - PCA.py : 주성분 분석을 통해 상권에서 상대적으로 높은 업종을 추천하기 위해 사용한 함수
        - relative_prevalence.py : flavorNetwork 논문을 기반으로 만든 알고리즘으로 상대비교를 통해 업종을 추천하는 함수
        - relative_prevalence_model_testFuc.py : flavorNetwork 함수를 만들기 위해 테스트한 함수
      
