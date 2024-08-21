
import pandas as pd
import numpy as np
import re
import holidays
us_holidays = holidays.US()

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, make_scorer
from sklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error, roc_auc_score
#pip install openfe
from openfe import OpenFE, tree_to_formula, transform

def clean_dataframe(df):
    # Drop problem columns
    problem_columns = df.select_dtypes(include=['datetime64[ns]', 'object']).columns
    df_cleaned = df.drop(columns=problem_columns, errors='ignore')
    
    # Convert timedelta64 columns to numeric values (in seconds)
    timedelta_columns = df_cleaned.select_dtypes(include=['timedelta64']).columns
    for col in timedelta_columns:
        df_cleaned[col] = df_cleaned[col].dt.total_seconds()

    df_cleaned = df_cleaned.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    return df_cleaned

def rmsle(y_true, y_pred):
    """Calculate the Root Mean Squared Logarithmic Error."""
    y_true = np.maximum(0, y_true) + 1
    y_pred = np.maximum(0, y_pred) + 1
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def cross_validated_scores_old(df_old, df_time, label, n_jobs, params, task_type):
    if task_type == 'regression':
        gbm = lgb.LGBMRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    elif task_type == 'classification':
        gbm = lgb.LGBMClassifier(**params)
        scoring = 'roc_auc'
    elif task_type == 'RMSLE':
        gbm = lgb.LGBMRegressor(**params)
        scoring = make_scorer(rmsle, greater_is_better=False) 
    
    cv_scores_old = cross_val_score(gbm, df_old, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    cv_scores_time = cross_val_score(gbm, df_time, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    
    if task_type in ['regression', 'RMSLE']:
        score_old = -cv_scores_old.mean()  # Convert to positive RMSE
        score_time = -cv_scores_time.mean()  # Convert to positive RMSE
    else:
        score_old = cv_scores_old.mean()
        score_time = cv_scores_time.mean()
    
    return score_old, score_time

def cross_validated_single(df, label, n_jobs, params, task_type):
    if task_type == 'regression':
        gbm = lgb.LGBMRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    elif task_type == 'classification':
        gbm = lgb.LGBMClassifier(**params)
        scoring = 'roc_auc'
    elif task_type == 'RMSLE':
        gbm = lgb.LGBMRegressor(**params)
        scoring = make_scorer(rmsle, greater_is_better=False) 

    if label.columns[0] in df.columns:
        df_copy = df#[label.columns[0]].copy()
        del df_copy[label.columns[0]]
    else:
        df_copy = df.copy()
    train_x, val_x, train_y, val_y = train_test_split(df, label, test_size=0.2, random_state=1)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    cv_scores= cross_val_score(gbm, df, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    
    if task_type in ['regression', 'RMSLE']:
        score = -cv_scores.mean()  # Convert to positive RMSE
    else:
        score = cv_scores.mean()
    
    return score

def save(list, file_path):
        with open(file_path, 'w') as file:
            file.write("Baselines ")
            file.write(list)
import pandas as pd
import time

def save_feature_importances_to_file(feature_importances, score, task_type, file_path):
        if feature_importances is None:
            print(f"Feature importances for {file_path} is None. Skipping saving.")
            return
        sorted_feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        with open(file_path, 'w') as file:
            file.write(str(score) + ' ' +task_type + ' ')
            file.write("Cross-validated feature importances before/after feature generation:\n")
            file.write(sorted_feature_importances.to_string())

def cross_validated_feature_importances_old(df, label, n_jobs, params, task_type):
    if task_type == 'regression':
        gbm = lgb.LGBMRegressor(**params)
    elif task_type == 'classification':
        gbm = lgb.LGBMClassifier(**params)
    elif task_type == 'RMSLE':
        gbm = lgb.LGBMRegressor(**params)
        #scoring = make_scorer(rmsle, greater_is_better=False)
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = df.columns
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_idx, val_idx in kf.split(df):
        train_x, val_x = df.iloc[train_idx], df.iloc[val_idx]
        train_y, val_y = label.iloc[train_idx], label.iloc[val_idx]
        gbm.fit(train_x, train_y)
        fold_importances = pd.DataFrame({'feature': df.columns, 'importance': gbm.feature_importances_})
        feature_importances = feature_importances.merge(fold_importances, on='feature', how='left', suffixes=(None, '_fold'))
    
    feature_importances['importance'] = feature_importances.drop('feature', axis=1).mean(axis=1)
    feature_importances = feature_importances[['feature', 'importance']]
    return feature_importances 

def cross_validated_scores(df_old, df_time, label, n_jobs, params, task_type):
    if task_type == 'regression':
        gbm = lgb.LGBMRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    elif task_type == 'classification':
        gbm = lgb.LGBMClassifier(**params)
        scoring = 'roc_auc'
    elif task_type == 'RMSLE':
        gbm = lgb.LGBMRegressor(**params)
        scoring = make_scorer(rmsle, greater_is_better=False) 
    # new
    if label.columns[0] in df_old.columns:
        df_old_copy = df_old#[label.columns[0]].copy()
        del df_old_copy[label.columns[0]]
    else:
        df_old_copy = df_old.copy()

    if label.columns[0] in df_time.columns:
        df_time_copy = df_time#[label.columns[0]].copy()
        del df_time_copy[label.columns[0]]
    else:
        df_time_copy = df_time.copy()
    # 
    train_x, val_x, train_y, val_y = train_test_split(df_old_copy, label, test_size=0.2, random_state=1)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    cv_scores_old = cross_val_score(gbm, df_old_copy, label, cv=10, scoring=scoring, n_jobs=n_jobs)

    train_x, val_x, train_y, val_y = train_test_split(df_time_copy, label, test_size=0.2, random_state=1)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    cv_scores_time = cross_val_score(gbm, df_time_copy, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    # old
    #cv_scores_old = cross_val_score(gbm, df_old, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    #cv_scores_time = cross_val_score(gbm, df_time, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    
    if task_type in ['regression', 'RMSLE']:
        score_old = -cv_scores_old.mean()  # Convert to positive RMSE
        score_time = -cv_scores_time.mean()  # Convert to positive RMSE
    else:
        score_old = cv_scores_old.mean()
        score_time = cv_scores_time.mean()
    
    return score_old, score_time


def cross_validated_feature_importances(df, label, n_jobs, params, task_type):
    if task_type == 'regression':
        gbm = lgb.LGBMRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    elif task_type == 'classification':
        gbm = lgb.LGBMClassifier(**params)
        scoring = 'roc_auc'
    elif task_type == 'RMSLE':
        gbm = lgb.LGBMRegressor(**params)
        scoring = make_scorer(rmsle, greater_is_better=False) 
    # new
    #train_x, val_x, train_y, val_y = train_test_split(df, label, test_size=0.2, random_state=1)
    #gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    #cv_scores_old = cross_val_score(gbm, df, label, cv=10, scoring=scoring, n_jobs=n_jobs)
    if label.columns[0] in df.columns:
        df_copy = df#[label.columns[0]].copy()
        del df_copy[label.columns[0]]
    else:
        df_copy = df.copy()
    #
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = df.columns
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_idx, val_idx in kf.split(df):
        train_x, val_x = df.iloc[train_idx], df.iloc[val_idx]
        train_y, val_y = label.iloc[train_idx], label.iloc[val_idx]
        #gbm.fit(train_x, train_y)
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
        fold_importances = pd.DataFrame({'feature': df.columns, 'importance': gbm.feature_importances_})
        feature_importances = feature_importances.merge(fold_importances, on='feature', how='left', suffixes=(None, '_fold'))
    
    feature_importances['importance'] = feature_importances.drop('feature', axis=1).mean(axis=1)
    feature_importances = feature_importances[['feature', 'importance']]
    return feature_importances

def prune(old_df, new_df, feature_importances, take_best):
    new_features = set(new_df.columns) - set(old_df.columns)
    
    new_feature_importances = feature_importances[feature_importances['feature'].isin(new_features)]

    sorted_importances = new_feature_importances.sort_values(by='importance', ascending=False)

    top_features = sorted_importances.head(take_best)['feature']

    pruned_new_df = new_df[top_features]

    top_features_list = top_features.tolist()
    
    return pruned_new_df, top_features_list

def prune(old_df, new_df, feature_importances, take_best):
    new_features = set(new_df.columns) - set(old_df.columns)
    new_feature_importances = feature_importances[feature_importances['feature'].isin(new_features)]
    sorted_importances = new_feature_importances.sort_values(by='importance', ascending=False)
    top_features = sorted_importances.head(take_best)['feature']
    old_features = old_df.columns
    selected_features = list(old_features) + top_features.tolist()
    pruned_new_df = new_df[selected_features]
    
    return pruned_new_df, selected_features

def compare_dataframes(df1, df2):
    common_attributes = df1.columns.intersection(df2.columns)
    unique_attributes_df1 = df1.columns.difference(df2.columns)
    unique_attributes_df2 = df2.columns.difference(df1.columns)
    
    return common_attributes, unique_attributes_df1, unique_attributes_df2


def run_last_stage(df_base, df_time_notpruned, flag, label_column, n_jobs, params, task_type, csv_file_path):
    #common_attributes, unique_attributes_df1, time_features = compare_dataframes(df_base, df_time_notpruned)
    print(label_column)
    import time
    time.sleep(5)
    #label = df_time_notpruned[[label_column]]

    #df_time_notpruned = df_time_notpruned.drop(columns=['dropoff_datetime__quarter']) # was needed?
    if flag == 2:
        id = 'id'
    elif flag == 3:
        id = 'TransactionID'
    else:
        df_base['TimeFE_ID'] = range(1, len(df_base) + 1)
        id = 'TimeFE_ID'

    label = df_time_notpruned[[label_column]].copy()
    df_time_notpruned = df_time_notpruned.drop(columns=[label_column])
    ofe = OpenFE()
    ofe.fit(data=df_time_notpruned, label=label, n_jobs=n_jobs)
    test_x = df_time_notpruned.copy()
    df_final, test_x = transform(df_time_notpruned, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)

    #ca, un, own_unique = compare_dataframes(df_base, df_time_notpruned)
    #pruned_time_df, kept_features = prune(df_base, df_time_notpruned, cross_validated_feature_importances(df_time_notpruned, label, n_jobs, params, task_type), 15)
    #columns_to_drop = [col for col in time_features if col not in kept_features and col not in common_attributes and col not in own_unique]
    #df_final = df_final.drop(columns=columns_to_drop, errors='ignore') 
    
    feature_importances_final = cross_validated_feature_importances(df_final, label, n_jobs, params, task_type)
    df_intact = df_base.copy()
    print(list(df_intact.columns))
    #common_attributes, unique_attributes_df1, time_features = compare_dataframes(df_intact, df_intact)
    #print(common_attributes)
    #df_base = df_base.drop(columns=[label_column])

    if label_column in df_base.columns: #new
        del df_base[label_column]

    df_final, kept_features = prune(df_base, df_final, feature_importances_final, 15)

    feature_importances_final = cross_validated_feature_importances(df_final, label, n_jobs, params, task_type)
    
    score_final = cross_validated_single(df_final, label, n_jobs, params, task_type)
    results = "for flag{}: ".format(flag) + " final score " + str([score_final]) + " " + task_type + " " + csv_file_path[4:]
    save(results, "Endresults/{}.txt".format(csv_file_path[4:len(csv_file_path)-4]))
    #save(feature_importances_final, "Endresults/end_importances{}.txt".format(csv_file_path[4:len(csv_file_path)-4]))
    save_feature_importances_to_file(feature_importances_final, score_final, task_type, "Endresults/end_importances{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))
    #
    df_final['TimeFE_ID'] = range(1, len(df_final) + 1)
    print(list(df_intact.columns))
    df_final = pd.merge(df_final, df_intact[[id,  label.columns[0]]], on=id, how='left', suffixes=('_new', '_old'))
    for col in df_final.columns:
        if col.endswith('_new'):
            original_col = col[:-4]
            old_col = f"{original_col}_old"
            if old_col in df_final.columns:
                # If columns are identical, drop the old one and rename the new one
                if df_final[col].equals(df_final[old_col]):
                    df_final.drop(columns=[old_col], inplace=True)
                    df_final.rename(columns={col: original_col}, inplace=True)
                else:
                    # If columns are not identical, keep both
                    df_final.rename(columns={col: original_col}, inplace=True)
    #
    df_final.to_csv('Endresults/df_final_{}.csv'.format(flag), index=False) 
    #final_prune, kept_features = prune(df_base, df_time_notpruned, cross_validated_feature_importances(df_time_notpruned, label, n_jobs, params, task_type), 15)
    



def run_experiment(flag, n_jobs):#, df_time_notpruned):
    results = ''

    print("### Processing CSV File {} ###".format(flag))
    #print(df_time_notpruned.columns.tolist())
    #return
    csv_files = [ 'data/NYC-Taxi.csv', 'data/NYC-BikeShare-2015-2017-combined.csv','data/retail_sales_dataset.csv', 
                 'data/NYCTraffic.csv', 'data/TravelTrip.csv', 'data/ChicagoTaxi.csv', 'data/retail_sales_dataset.csv', 
                 'data/online_retail_II.csv', 'data/retail_opti.csv' ,'data/RidesChicago.csv', 'data/Yellow_Taxi.csv', 
                 'data/SeoulBikeData.csv', 'data/seattle-weather.csv', 'data/austin_weather.csv', 
                 'data/london_weather.csv', 'data/CarSales.csv', 'data/supermarket_sales.csv', 'data/Employee.csv' ]
    if flag == 1: 
        csv_file_path = 'data/NYC-Taxi.csv'
        label_column = 'trip_duration'
    elif flag == 2: #not working
        csv_file_path = 'data/NYC-BikeShare-2015-2017-combined.csv'
        label_column = 'Gender'
    elif flag == 3: #not working, now working (:
        csv_file_path = 'data/retail_sales_dataset.csv'
        label_column = 'Total_Amount'
    elif flag == 4:
        csv_file_path = 'data/NYCTraffic.csv' 
        label_column = 'value' #own label
    elif flag == 5: #not working
        csv_file_path = 'data/online_retail_II.csv'
        label_column = 'Price' #own label lag_price
    elif flag == 6:#not working
        csv_file_path = 'data/retail_opti.csv'
        label_column = 'lag_price'
    elif flag == 7:
        csv_file_path = 'data/TravelTrip.csv'
        label_column = 'Traveler_gender' #own label 
    elif flag == 8:
        csv_file_path = 'data/ChicagoTaxi.csv'
        label_column = 'Fare'
    elif flag == 9:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'lag_price'
    elif flag == 11: #rideschicago
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path) 
        label_column = 'fare_amount'
    elif flag == 10: #yellowtaxi
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'fare_amount'
    elif flag == 12:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'Rented_Bike_Count'
    elif flag == 13:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'weather'
        df['weather'] = le.fit_transform(df['weather'])
    elif flag == 14: #not working
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path).dropna()
        df['HumidityAvgPercent'] = pd.to_numeric(df['HumidityAvgPercent'], errors='coerce')
        df = df.dropna(subset=['HumidityAvgPercent'])
        df['HumidityAvgPercent'] = df['HumidityAvgPercent'].astype(int)
        label_column ='HumidityAvgPercent'
    elif flag == 15:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'precipitation'
    elif flag == 16:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'Price'
    elif flag == 17:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'Total'
        gender_mapping = {'Male': 1, 'Female': 0}
        df['Gender'] = df['Gender'].replace(gender_mapping)
        Customer_mapping = {'Member': 1, 'Normal': 0} 
        df['Customer_type'] = df['Customer_type'].replace(gender_mapping)
    elif flag == 18:
        csv_file_path = csv_files[flag-1]
        df = pd.read_csv(csv_file_path)
        label_column = 'LeaveOrNot'
        gender_mapping = {'Male': 1, 'Female': 0}
        df['Gender'] = df['Gender'].replace(gender_mapping)
    else:
        raise ValueError("Invalid flag value")

    # Load and preprocess the dataframe
    if flag <= 8:
        df = pd.read_csv(csv_file_path)
    if flag == 3:
        gender_mapping = {'Male': 1, 'Female': 0}
        df['Gender'] = df['Gender'].replace(gender_mapping)

    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()


    #print(df.columns.tolist())
    #return
    # Create data and label
    label = df[[label_column]]
    #df_time_notpruned[label_column] = 
    data = df.drop(columns=[label_column]).copy()
    if flag == 14:
        data = df.drop(columns=['Events']).copy()
        df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))
        df = df.select_dtypes(include=[np.number])
        df = df.dropna()
        if df.isna().any().any():
            raise ValueError("Data contains NaNs")
        print("scaled")
    #print(data.columns.tolist())
    print(label)
    import time
    time.sleep(5)

    
    ofe = OpenFE()
    start_ofe = time.time()
    ofe.fit(data=data, label=label, n_jobs=n_jobs)
    test_x = df.copy()
    # df_open_train, df_open_test = train_test_split(df_open, label, test_size=0.2, random_state=1)
    df_open, test_x = transform(data.copy(), test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    end_ofe = time.time()
    overhead = end_ofe - start_ofe
    df_old = df.drop(columns=[label_column]).copy()
    df_intact = df.copy()
    df_old = clean_dataframe(df_old)
    df_intact = clean_dataframe(df_intact)

    df_open = clean_dataframe(df_open).copy()
    #df_open = df_open.drop(columns=[label_column]).copy()
    
    if flag == 2:
        task_type = 'RMSLE'
    else:
        task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'

    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    
    feature_importances_old_cv = cross_validated_feature_importances(df_old.copy(), label, n_jobs, params, task_type)
    #save(feature_importances_old_cv, "Baselines/featimp{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))
    feature_importances_time_cv = cross_validated_feature_importances(df_open.copy(), label, n_jobs, params, task_type)
    score_old, score_time = cross_validated_scores(df_old.copy(), df_open.copy(), label, n_jobs, params, task_type)
    
    results = "for flag{}: ".format(flag) + " score old/new " + str([score_old, score_time]) + " " + task_type + " " + csv_file_path[5:]
    
    def save_feature_importances_to_file(feature_importances, score, task_type, file_path):
        if feature_importances is None:
            print(f"Feature importances for {file_path} is None. Skipping saving.")

        sorted_feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        with open(file_path, 'w') as file:
            file.write(str(score) + ' ' +task_type + ' ')
            file.write("Cross-validated feature importances before/after feature generation:\n")
            file.write(sorted_feature_importances.to_string())

    save(results, "Baselines/{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))

    save_feature_importances_to_file(feature_importances_old_cv, score_old, task_type, "Baselines/base_importances{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))
    save_feature_importances_to_file(feature_importances_time_cv, score_time, task_type, "Baselines/openfe_importances{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))
    '''
    label = df_time_notpruned[[label_column]]
    df_time_notpruned = df_time_notpruned.drop(columns=[label_column]).copy()
    score_notpruned = cross_validated_single(df_time_notpruned, label, n_jobs, params, task_type)
    feature_importances_notpruned_cv = cross_validated_feature_importances(df_time_notpruned.copy(), label, n_jobs, params, task_type)
    save_feature_importances_to_file(feature_importances_notpruned_cv, score_notpruned, task_type, "Baselines/not_pruned_importance{}.txt".format(csv_file_path[5:len(csv_file_path)-4]))
    '''
    #df_time_notpruned = df_time_notpruned.sample(frac=0.3, random_state=1).dropna()
    #df_intact = df_intact.sample(frac=0.3, random_state=1).dropna()
    #run_last_stage(df_intact, df_time_notpruned, flag, label_column, n_jobs, params, task_type, csv_file_path)

'''
def run_last_stage(df_base, df_time_notpruned, flag, label_column, n_jobs, params, task_type, csv_file_path):
    print(f"Running last stage with label column: {label_column}")
    
    # Check if the label column exists in df_time_notpruned
    if label_column not in df_time_notpruned.columns:
        raise ValueError(f"Label column '{label_column}' not found in df_time_notpruned")

    label = df_time_notpruned[[label_column]].copy()  # Ensure label is a DataFrame
    df_time_notpruned = df_time_notpruned.drop(columns=[label_column])
    
    ofe = OpenFE()
    ofe.fit(data=df_time_notpruned, label=label, n_jobs=n_jobs)
    
    test_x = df_time_notpruned.copy()
    df_final, test_x = transform(df_time_notpruned, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    
    feature_importances_final = cross_validated_feature_importances(df_final, label, n_jobs, params, task_type)
    df_final, kept_features = prune(df_base, df_final, feature_importances_final, 15)
    score_final = cross_validated_single(df_final, label, n_jobs, params, task_type)
    
    results = f"for flag{flag}: final score {score_final} {task_type} {csv_file_path[4:]}"
    save(results, f"Endresults/{csv_file_path[4:-4]}.txt")
    save(feature_importances_final, f"Endresults/end_importances{csv_file_path[4:-4]}.txt")

def run_experiment(flag, n_jobs, df_time_notpruned):
    results = ''
    print(f"### Processing CSV File {flag} ###")
    
    label_column = ''
    csv_file_path = ''
    
    if flag == 1: 
        csv_file_path = 'data/NYC-Taxi.csv'
        label_column = 'trip_duration'
    elif flag == 2:
        csv_file_path = 'data/NYC-BikeShare-2015-2017-combined.csv'
        label_column = 'Gender'
    elif flag == 3:
        csv_file_path = 'data/retail_sales_dataset.csv'
        label_column = 'Total_Amount'
    elif flag == 4:
        csv_file_path = 'data/NYCTraffic.csv' 
        label_column = 'value'
    elif flag == 5:
        csv_file_path = 'data/online_retail_II.csv'
        label_column = 'Price'
    elif flag == 6:
        csv_file_path = 'data/retail_opti.csv'
        label_column = 'lag_price'
    elif flag == 7:
        csv_file_path = 'data/TravelTrip.csv'
        label_column = 'Traveler_gender'
    elif flag == 8:
        csv_file_path = 'data/ChicagoTaxi.csv'
        label_column = 'Fare'
    else:
        raise ValueError("Invalid flag value")

    # Load and preprocess the dataframe
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.02, random_state=1).dropna()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Check if label_column exists in the DataFrame
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset loaded from {csv_file_path}")

    # Create data and label
    label = df[[label_column]].copy()
    data = df.drop(columns=[label_column])
    
    ofe = OpenFE()
    start_ofe = time.time()
    ofe.fit(data=data, label=label, n_jobs=n_jobs)
    test_x = df.copy()
    df_open, test_x = transform(df, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    end_ofe = time.time()
    overhead = end_ofe - start_ofe

    df_old = df.drop(columns=[label_column]).copy()
    df_intact = df.copy()
    df_old = clean_dataframe(df_old)
    df_intact = clean_dataframe(df_intact)
    df_open = clean_dataframe(df_open)
    
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    
    feature_importances_old_cv = cross_validated_feature_importances(df_old, label, n_jobs, params, task_type)
    feature_importances_time_cv = cross_validated_feature_importances(df_open, label, n_jobs, params, task_type)
    score_old, score_time = cross_validated_scores(df_old, df_open, label, n_jobs, params, task_type)
    
    print(f"Columns in df_time_notpruned before run_last_stage: {df_time_notpruned.columns.tolist()}")
    run_last_stage(df_intact, df_time_notpruned, flag, label_column, n_jobs, params, task_type, csv_file_path)
'''

def rename_columns_except_label(df, label):
    # Define the renaming function with a conditional check for the label
    def rename_column(col):
        if col == label:
            return col
        else:
            return re.sub('[^A-Za-z0-9_]+', '', col)
    
    # Rename columns using the conditional renaming function
    df = df.rename(columns=rename_column).copy()
    return df

# Entry point for the script


def match_sampled_dfs(df1, df2, id_column, frac, random_state=None):
    """
    Sample df2 to match the rows in df1 based on the id_column.

    Parameters:
    df1 (pd.DataFrame): DataFrame that was previously sampled.
    df2 (pd.DataFrame): Original DataFrame to be sampled.
    id_column (str): Column name to match rows.
    frac (float): Fraction of rows to sample.
    random_state (int, optional): Random state for reproducibility.

    Returns:
    pd.DataFrame: Sampled DataFrame (df2) with the same rows as df1.
    """
    sampled_ids = df1[id_column].unique()
    matched_df2 = df2[df2[id_column].isin(sampled_ids)].copy()
    df1_sampled = df1.sample(frac=frac, random_state=1)
    matched_df2_sampled = matched_df2.sample(frac=frac, random_state=1)
    
    return df1_sampled, matched_df2_sampled

if __name__ == '__main__':
    n_jobs = 8  
    #flag = 7
    #while flag < 9:
        #run_experiment(flag, n_jobs)
        #flag += 1
    #run_experiment(2, n_jobs)

    csv_files = [ 'data/NYC-Taxi.csv', 'data/NYC-BikeShare-2015-2017-combined.csv','data/retail_sales_dataset.csv', 
                 'data/NYCTraffic.csv', 'data/TravelTrip.csv', 'data/ChicagoTaxi.csv', 'data/retail_sales_dataset.csv', 
                 'data/online_retail_II.csv', 'data/retail_opti.csv' ,'data/RidesChicago.csv', 'data/Yellow_Taxi.csv', 
                 'data/SeoulBikeData.csv', 'data/seattle-weather.csv', 'data/austin_weather.csv', 
                 'data/london_weather.csv', 'data/CarSales.csv', 'data/supermarket_sales.csv', 'data/Employee.csv' ]
    label_columns = [
    'trip_duration',           # flag == 1
    'Gender',                  # flag == 2
    'Total_Amount',            # flag == 3
    'value',                   # flag == 4
    'Price',                   # flag == 5
    'lag_price',               # flag == 6
    'Traveler_gender',         # flag == 7
    'Fare',                    # flag == 8
    'lag_price',               # flag == 9
    'fare_amount',             # flag == 10
    'fare_amount',             # flag == 11
    'Rented_Bike_Count',       # flag == 12
    'weather',                 # flag == 13
    'HumidityAvgPercent',      # flag == 14
    'precipitation',           # flag == 15
    'Price',                   # flag == 16
    'Total',                   # flag == 17
    'LeaveOrNot'               # flag == 18
]

    for flag in range(1, 19):
        df_time_notpruned = pd.read_csv('Data/Importances/same_new_df_{0}.csv'.format(flag))
        df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
        print(df_time_notpruned.head())
        import time
        time.sleep(2)
        csv_file_path = csv_files[flag-1]
        #if flag > 16:
        run_experiment(flag, 8)
        label_column = label_columns[flag-1]
        df = pd.read_csv(csv_file_path)
        df = df.sample(frac=0.8, random_state=1).dropna()
        df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
        df_intact = clean_dataframe(df)
        params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
        label = df[[label_column]]
        task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
        run_last_stage(df_intact, df_time_notpruned, flag, label_column, n_jobs, params, task_type, csv_file_path)

    import time
    time.sleep(2)
    '''df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_1.csv')
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    #if 'trip_duration' not in df_time_notpruned.columns:
    #    raise ValueError("The label column 'trip_duration' does not exist in df_time_notpruned")

    run_experiment(1, n_jobs, df_time_notpruned)

    csv_file_path = 'data/NYC-Taxi.csv'
    label_column = 'trip_duration'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 1, label_column, n_jobs, params, task_type, csv_file_path)
    
    #df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_2.csv')
    df_time_notpruned =  pd.read_csv('Data/Importances/pruned_df_2.csv')
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    df_time_notpruned = df_time_notpruned.sample(frac=0.3, random_state=1).dropna()
    csv_file_path = 'data/NYC-BikeShare-2015-2017-combined.csv'
    label_column = 'Gender'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.3, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    #df_intact = df_intact.sample(frac=0.3, random_state=1).dropna()
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    df_intact, df_time_notpruned = match_sampled_dfs(df_intact, df_time_notpruned, 'id', frac=0.3, random_state=1)
    run_last_stage(df_intact, df_time_notpruned, 2, label_column, n_jobs, params, 'RMSLE', csv_file_path)
    #run_experiment(2, n_jobs)

    #run_experiment(3, n_jobs)
    df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_7.csv') #not 3
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    #df_time_notpruned = df_time_notpruned.sample(frac=0.6, random_state=1).dropna()
    csv_file_path = 'data/retail_sales_dataset.csv'
    label_column = 'Total_Amount'
    df = pd.read_csv(csv_file_path)
    gender_mapping = {'Male': 1, 'Female': 0}
    df['Gender'] = df['Gender'].replace(gender_mapping)
    df = df.sample(frac=0.8, random_state=1).dropna()
    #df = df.sample(frac=0.8 * 0.6, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    #df_intact, df_time_notpruned = match_sampled_dfs(df_intact, df_time_notpruned, 'TransactionID', frac=0.3, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df_intact[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 3, label_column, n_jobs, params, task_type, csv_file_path)'''
    #run_experiment(3, n_jobs)

    df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_8.csv')
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    csv_file_path = 'data/online_retail_II.csv'
    label_column = 'Price' #own label lag_price
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 5, label_column, n_jobs, params, task_type, csv_file_path)
    #run_experiment(5, n_jobs)

    df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_9.csv')
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    csv_file_path = 'data/retail_opti.csv'
    label_column = 'lag_price'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 6, label_column, n_jobs, params, task_type, csv_file_path)
    #run_experiment(6, n_jobs)

    ###
    '''
    df_time_notpruned = pd.read_csv('Data/Importances/same_new_df_4.csv')
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    #if 'value' not in df_time_notpruned.columns:
    #    raise ValueError("The label column 'trip_duration' does not exist in df_time_notpruned")
    
    run_experiment(4, n_jobs, df_time_notpruned)

    csv_file_path = 'data/NYCTraffic.csv' 
    label_column = 'value'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 8, label_column, n_jobs, params, task_type, csv_file_path)

    ###

    df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_5.csv') #not 7!
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    #if 'trip_duration' not in df_time_notpruned.columns:
    #    raise ValueError("The label column 'trip_duration' does not exist in df_time_notpruned")
    
    #run_experiment(7, n_jobs, df_time_notpruned)

    csv_file_path = 'data/TravelTrip.csv'
    label_column = 'Traveler_gender'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    #df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df = rename_columns_except_label(df, label_column)
    if 'Traveler_gender' not in df.columns:
        raise ValueError("The label column 'trip_duration' does not exist in df_time_notpruned")
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 8, label_column, n_jobs, params, task_type, csv_file_path)

    ###
    
    df_time_notpruned =  pd.read_csv('Data/Importances/same_new_df_6.csv').dropna() #not 8!
    df_time_notpruned = df_time_notpruned.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    
    #if 'trip_duration' not in df_time_notpruned.columns:
    #    raise ValueError("The label column 'trip_duration' does not exist in df_time_notpruned") 

    run_experiment(8, n_jobs, df_time_notpruned)

    csv_file_path = 'data/ChicagoTaxi.csv'
    label_column = 'Fare'
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.8, random_state=1).dropna()
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x)).copy()
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x)).copy()
    df_intact = clean_dataframe(df)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
    label = df[[label_column]]
    task_type = 'classification' if label.nunique().values[0] == 2 else 'regression'
    run_last_stage(df_intact, df_time_notpruned, 8, label_column, n_jobs, params, task_type, csv_file_path)'''