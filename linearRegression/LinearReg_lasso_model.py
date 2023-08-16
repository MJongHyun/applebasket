import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from regressors import stats

# 데이터가 주어졌을 때, Lasso모델에 alpha값을 적용하여, 가장 RMSE가 작은 값을 모델로 추출 및 결과값들 추출

def best_lasso_model(X_train, y_train, X_test, y_test, alpha_list = False):  # 입력값: Train/Test 값, alpha값들 추출

    # alpha 값들을 정하지 않을 경우, 함수 안에 있는 alpha_list로 진행
    if not alpha_list:

        alpha_list = [5e-05, 1e-05, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100]

    else:

        alpha_list = alpha_list

    RMSE = 9999999999

    for i in alpha_list:

        lasso_reg = Lasso(alpha=i, normalize=True)
        lasso_reg.fit(X_train, y_train)

        y_pred = lasso_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        if rmse <= RMSE:
            RMSE = rmse
            coef = lasso_reg.coef_
            R2 = r2_score(y_test, y_pred)
            best_alpha = i

    coef_df = pd.DataFrame(coef).T
    coef_df.columns = X_train.columns
    coef_df['RMSE'] = RMSE
    coef_df['COEF'] = R2
    coef_df['Alpha'] = best_alpha

    return coef_df, lasso_reg


def best_lasso_model2(X_train, y_train, X_test, y_test, alpha_list = False):  # 입력값: Train/Test 값, alpha값들 추출

    # alpha 값들을 정하지 않을 경우, 함수 안에 있는 alpha_list로 진행
    if not alpha_list:

        alpha_list = [5e-05, 1e-05, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100]

    else:

        alpha_list = alpha_list

    while 1:

        alpha_list = [5e-05, 1e-05, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100]

        RMSE = 9999999999

        for i in alpha_list:

            lasso_reg = Lasso(alpha = i, normalize = True)
            lasso_reg.fit(X_train, y_train)

            y_pred = lasso_reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            if rmse <= RMSE:
                RMSE = rmse
                coef = lasso_reg.coef_
                R2 = r2_score(y_test, y_pred)
                best_alpha = i

        # coeff가 0보다 큰 경우, 값을 제외하고, 다시 회귀분석을 진행하여 최적화 된 독립변수를 찾기

        check = pd.DataFrame(coef).T
        check.columns = X_train.columns

        is_zero = check[check <= 0].T.dropna().index

        if len(is_zero) > 0:
            X_train = X_train.drop(is_zero, axis = 'columns')
            X_test = X_test.drop(is_zero, axis = 'columns')

        else:

            check['RMSE'] = RMSE
            check['COEF'] = R2
            check['Alpha'] = best_alpha
            break

    return check

# 모델에 따른, 컬럼별 P-value 확인하기

def check_p_val(reg_model, X_train, y_train):

    check_coef_pval = stats.coef_pval(reg_model, X_train, y_train)
    stats.summary(reg_model, X_train, y_train)

    return check_coef_pval