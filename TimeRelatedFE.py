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

def detect_mostly_unique_and_convert(df, threshold=0.98, base_date='2020-01-01', max_days=36500):
    mostly_unique = []
    base_date = pd.to_datetime(base_date)
    
    for column in df.columns:
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio >= threshold:
            mostly_unique.append(column)
            # Map unique values to sequential dates starting from the base_date
            unique_values = df[column].unique()
            
            # Ensure we do not exceed the max_days limit
            if len(unique_values) > max_days:
                unique_values = unique_values[:max_days]
            
            date_mapping = {val: base_date + pd.Timedelta(days=i) for i, val in enumerate(np.sort(unique_values))}
            df[f'{column}_$_datemap'] = df[column].map(date_mapping)
    
    return df, mostly_unique

def detect_mostly_not_unique_and_convert(df, threshold=0.10):
    mostly_not_unique = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_map = {day: i for i, day in enumerate(weekdays)}
    
    for column in df.columns:
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio <= threshold:
            mostly_not_unique.append(column)
            # Check if the column values match the weekday pattern or other specific patterns
            if set(df[column].unique()).issubset(set(weekdays)):
                 df[f'{column}_$_weekmap'] = df[column].map(weekday_map)
            else:
                # General conversion to category codes for other categorical columns
                df[f'{column}_$_category'] = df[column].astype('category').cat.codes
    
    return df, mostly_not_unique

def detect_time_increasing(df):
    time_increasing = []
    for column in df.columns:
        if df[column].is_monotonic_increasing:
            time_increasing.append(column)
    return time_increasing

def ordinally_encode_dataframe(df):
    encoded_df = df.copy()
    for column in df.columns:
        # Handle datetime columns separately
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            encoded_df[f'{column}_$_ordinal'] = df[column].rank(method='dense').fillna(-1).astype(int)
        # Handle categorical columns separately
        elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
            encoded_df[f'{column}_$_ordinal'] = pd.factorize(df[column])[0] + 1
            encoded_df[f'{column}_$_ordinal'] = encoded_df[f'{column}_$_ordinal'].replace(-1, 0)
        # Handle numerical columns separately
        else:
            encoded_df[f'{column}_$_ordinal'] = df[column].rank(method='dense').fillna(-1).astype(int)
    return encoded_df

def TimeRelatedFe(df, holiday_countries=['us'], date_format='%Y-%m-%d %H:%M:%S'): #'%Y-%m-%d %H:%M:%S'

    if holiday_countries[0] == 'us':
        us_holidays = holidays.US()
    # Detect time increasing features
    time_increasing = detect_time_increasing(df)
    print("Time Increasing Features:", time_increasing)
    
    def is_datetime_column(column):
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%Y%m%d'
        ]
        
        if pd.api.types.is_datetime64_any_dtype(column):
            return True
        
        if pd.api.types.is_object_dtype(column):
            for date_format in formats:
                try:
                    # Attempt to convert to datetime
                    converted = pd.to_datetime(column, format=date_format, errors='raise')
                    # Check if conversion was successful
                    if pd.api.types.is_datetime64_any_dtype(converted):
                        # Change the column to the specified datetime format
                        column[:] = converted.dt.strftime('%Y-%m-%d %H:%M:%S')
                        return True
                except (ValueError, TypeError):
                    continue  # Try the next format
        
        return False

    def convert_datetime_columns(df):
        new_columns = []
        datetime_columns = []

        for column in df.columns:
            if is_datetime_column(df[column]):
                df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
                datetime_columns.append(column)
                df[f'{column}_$_year'] = df[column].dt.year
                df[f'{column}_$_month'] = df[column].dt.month
                df[f'{column}_$_day'] = df[column].dt.day
                df[f'{column}_$_hour'] = df[column].dt.hour
                df[f'{column}_$_minute'] = df[column].dt.minute
                df[f'{column}_$_second'] = df[column].dt.second
                df[f'{column}_$_day_of_week'] = df[column].dt.dayofweek
                df[f'{column}_$_day_name'] = df[column].dt.day_name()
                df[f'{column}_$_month_name'] = df[column].dt.month_name()
                df[f'{column}_$_week_of_year'] = df[column].dt.isocalendar().week
                df[f'{column}_$_is_weekend'] = df[column].dt.dayofweek >= 5
                df[f'{column}_$_is_year_start'] = df[column].dt.month.isin([1, 2, 3])
                df[f'{column}_$_is_year_end'] = df[column].dt.month.isin([10, 11, 12])
                df[f'{column}_$_quarter'] = df[column].dt.quarter
                
                df[f'{column}_$_is_holiday'] = df[column].apply(lambda x: True if x.date() in us_holidays else False) #changed bools
                
                def days_to_nearest_holiday(date):
                    if pd.notna(date):
                        days_diff = [abs((date.date() - pd.Timestamp(holiday).date()).days) for holiday in us_holidays]
                        return min(days_diff) if days_diff else None
                    return None
                
                df[f'{column}_$_days_to_nearest_holiday'] = df[column].apply(days_to_nearest_holiday)

                def check_date_within_range(date, holiday_list, days_range):
                    from datetime import timedelta
                    if pd.isna(date):
                        return False

                    # Calculate start_date and end_date based on days_range
                    start_date = date + timedelta(days=days_range)
                    end_date = date

                    # If days_range is positive, we should be looking after the reference date
                    if days_range >= 0:
                        start_date = date
                        end_date = date + timedelta(days=days_range)
                    
                    # If days_range is negative, we should be looking before the reference date
                    elif days_range < 0:
                        start_date = date + timedelta(days=days_range)
                        end_date = date

                    # Check if any date in the range is a holiday
                    for single_date in pd.date_range(start=start_date, end=end_date):
                        if single_date.date() in holiday_list:
                            return True
                    return False

                # Add columns for different ranges around holidays
                df[f'{column}_$_within_1_week_before_holiday'] = df[column].apply(lambda x: check_date_within_range(x, us_holidays, days_range=-7))
                df[f'{column}_$_within_1_month_before_holiday'] = df[column].apply(lambda x: check_date_within_range(x, us_holidays, days_range=-30))
                df[f'{column}_$_within_1_week_after_holiday'] = df[column].apply(lambda x: check_date_within_range(x, us_holidays, days_range=7))
                df[f'{column}_$_within_1_month_after_holiday'] = df[column].apply(lambda x: check_date_within_range(x, us_holidays, days_range=30))

                


                new_columns.extend([
                    f'{column}_$_year', f'{column}_$_month', f'{column}_$_day', f'{column}_$_hour',
                    f'{column}_$_minute', f'{column}_$_second', f'{column}_$_day_of_week',
                    f'{column}_$_day_name', f'{column}_$_month_name', f'{column}_$_week_of_year',
                    f'{column}_$_is_holiday', f'{column}_$_days_to_nearest_holiday'
                ])

        return df, new_columns, datetime_columns

    def add_trigonometric_columns(df, datetime_columns):
        new_columns = []

        for column in datetime_columns:
            df[f'{column}_$_month_sin'] = np.sin(2 * np.pi * df[column].dt.month / 12)
            df[f'{column}_$_month_cos'] = np.cos(2 * np.pi * df[column].dt.month / 12)

            df[f'{column}_$_day_sin'] = np.sin(2 * np.pi * df[column].dt.day / 31)
            df[f'{column}_$_day_cos'] = np.cos(2 * np.pi * df[column].dt.day / 31)

            df[f'{column}_$_hour_sin'] = np.sin(2 * np.pi * df[column].dt.hour / 24)
            df[f'{column}_$_hour_cos'] = np.cos(2 * np.pi * df[column].dt.hour / 24)

            new_columns.extend([
                f'{column}_$_month_sin', f'{column}_$_month_cos',
                f'{column}_$_day_sin', f'{column}_$_day_cos',
                f'{column}_$_hour_sin', f'{column}_$_hour_cos'
            ])

        return df, new_columns

    def calculate_time_differences(df, datetime_columns):
        time_difference_columns = []
        same_day_columns = []

        for i in range(len(datetime_columns)):
            for j in range(i + 1, len(datetime_columns)):
                col1 = datetime_columns[i]
                col2 = datetime_columns[j]
                diff_col_name = f'_$_{col1}_vs_$_{col2}_timedelta'
                df[diff_col_name] = (df[col1] - df[col2]).abs()
                time_difference_columns.append(diff_col_name)

                same_day_col_name = f'_$_{col1}_vs_$_{col2}_same_day'
                df[same_day_col_name] = (df[col1].dt.date == df[col2].dt.date).astype(int)
                same_day_columns.append(same_day_col_name)

        return df, time_difference_columns, same_day_columns

    def clean_column_names(df):
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        return df
    
    df, mostly_not_unique = detect_mostly_not_unique_and_convert(df, threshold=0.10)
    print("Mostly Not Unique Features:", mostly_not_unique)
    df, mostly_unique = detect_mostly_unique_and_convert(df, threshold=0.98)
    print("Mostly Unique Features:", mostly_unique)


    df, new_columns, datetime_columns = convert_datetime_columns(df)
    df, trigonometric_columns = add_trigonometric_columns(df, datetime_columns)
    df, time_difference_columns, same_day_columns = calculate_time_differences(df, datetime_columns)
    df = clean_column_names(df)
    df = ordinally_encode_dataframe(df)

    return df


#################

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

def get_score(train_x, test_x, train_y, test_y, n_jobs):
    timedelta_columns = train_x.select_dtypes(include=['timedelta64']).columns
    for col in timedelta_columns:
        train_x[col] = train_x[col].dt.total_seconds()
        test_x[col] = test_x[col].dt.total_seconds()
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1, 'verbose': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    feature_importances = pd.DataFrame({'feature': gbm.feature_name_, 'importance': gbm.feature_importances_})
    return score, feature_importances

def compare_dataframes(df1, df2):
    common_attributes = df1.columns.intersection(df2.columns)
    unique_attributes_df1 = df1.columns.difference(df2.columns)
    unique_attributes_df2 = df2.columns.difference(df1.columns)
    
    return common_attributes, unique_attributes_df1, unique_attributes_df2



# write your scorers here #

def rmsle(y_true, y_pred):
    """Calculate the Root Mean Squared Logarithmic Error."""
    y_true = np.maximum(0, y_true) + 1
    y_pred = np.maximum(0, y_pred) + 1
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# write your scorers here #

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
    train_x, val_x, train_y, val_y = train_test_split(df_old, label, test_size=0.2, random_state=1)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    cv_scores_old = cross_val_score(gbm, df_old, label, cv=10, scoring=scoring, n_jobs=n_jobs)

    train_x, val_x, train_y, val_y = train_test_split(df_time, label, test_size=0.2, random_state=1)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
    cv_scores_time = cross_val_score(gbm, df_time, label, cv=10, scoring=scoring, n_jobs=n_jobs)
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
    if label.columns[0] in df.columns:
        df_copy = df
        del df_copy[label.columns[0]]
    else:
        df_copy = df.copy()
    #
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = df.columns
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_idx, val_idx in kf.split(df_copy):
        train_x, val_x = df_copy.iloc[train_idx], df_copy.iloc[val_idx]
        train_y, val_y = label.iloc[train_idx], label.iloc[val_idx]
    
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
        fold_importances = pd.DataFrame({'feature': df_copy.columns, 'importance': gbm.feature_importances_})
        feature_importances = feature_importances.merge(fold_importances, on='feature', how='left', suffixes=(None, '_fold'))
    
    feature_importances['importance'] = feature_importances.drop('feature', axis=1).mean(axis=1)
    feature_importances = feature_importances[['feature', 'importance']]
    return feature_importances

##################


def read_feature_importances(file_path):
    # Read the file with proper separator and handle extra header lines
    df = pd.read_csv(file_path, sep=r'\s{2,}', engine='python', skiprows=1)
    # Remove any leading or trailing whitespace in column names
    df.columns = df.columns.str.strip()
    return df

def extract_suffix(feature_name):
    parts = feature_name.split('__')
    if len(parts) > 1:
        return '__' + parts[-1]
    return ''

def calculate_improvements(new_file_path, output_file):
    feature_importances = read_feature_importances(new_file_path)
    
    improvements = []
    transformation_performance = {}

    # Old features are those without the '__' suffix
    old_features = [f for f in feature_importances['feature'] if '__' not in f]
    
    for old_feature in old_features:
        old_importance = feature_importances[feature_importances['feature'] == old_feature]['importance'].values[0]
        
        # New rows derived from old features
        new_rows = feature_importances[feature_importances['feature'].str.contains(f'{old_feature}__')]
        
        for _, new_row in new_rows.iterrows():
            new_feature = new_row['feature']
            new_importance = new_row['importance']
            improvement = ((new_importance - old_importance) / old_importance) * 100
            
            improvements.append({
                'old_feature': old_feature,
                'new_feature': new_feature,
                'old_importance': old_importance,
                'new_importance': new_importance,
                'improvement (%)': improvement
            })
            
            suffix = extract_suffix(new_feature)
            if suffix not in transformation_performance:
                transformation_performance[suffix] = []
            transformation_performance[suffix].append(improvement)
    
    # Aggregate improvements by suffix
    transformation_summary = []
    for suffix, values in transformation_performance.items():
        avg_improvement = sum(values) / len(values) if values else 0
        max_improvement = max(values) if values else 0
        min_improvement = min(values) if values else 0
        transformation_summary.append({
            'transformation': suffix,
            'average_improvement (%)': avg_improvement,
            'max_improvement (%)': max_improvement,
            'min_improvement (%)': min_improvement
        })
    
    # Save results to CSV
    improvements_df = pd.DataFrame(improvements)
    transformation_summary_df = pd.DataFrame(transformation_summary)
    
    improvements_df.to_csv(output_file.replace('.csv', '_improvements.csv'), index=False)
    transformation_summary_df.to_csv(output_file.replace('.csv', '_summary.csv'), index=False)


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

if __name__ == "__main__":
    import pandas as pd

    def save_feature_importances_to_file(feature_importances, score, task_type, file_path):
        sorted_feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        with open(file_path, 'w') as file:
            file.write(str(score) + ' ' +task_type + ' ')
            file.write("Cross-validated feature importances before/after feature generation:\n")
            file.write(sorted_feature_importances.to_string())

    csv_files = [ 'data/NYC-Taxi.csv', 'data/NYC-BikeShare-2015-2017-combined.csv','data/retail_sales_dataset.csv', 
                 'data/NYCTraffic.csv', 'data/TravelTrip.csv', 'data/ChicagoTaxi.csv', 'data/retail_sales_dataset.csv', 
                 'data/online_retail_II.csv', 'data/retail_opti.csv' , 'data/Yellow_Taxi.csv', 'data/RidesChicago.csv' 
                 'data/SeoulBikeData.csv', 'data/seattle-weather.csv', 'data/austin_weather.csv', 
                 'data/london_weather.csv', 'data/CarSales.csv', 'data/supermarket_sales.csv', 'data/Employee.csv' ]
    n_jobs = 8

    for flag, csv_file_path in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_file_path).dropna() 
        df = df.sample(frac=0.8, random_state=11)
        df['TimeFE_ID'] = range(1, len(df) + 1)
        data = df.copy()

        print("### Processing CSV File {} ###".format(flag))
        print(csv_file_path)
        if flag == 2: #war 1
            label = data[['Gender']]
            del data['Gender']
        elif flag == 1: #war 2
            id = 'vendor_id'
            label = data[['trip_duration']]
            del data['trip_duration']
        elif flag == 3: #need delete
            label = data[['Total_Amount']]
            del data['Total_Amount']
        elif flag == 4:
            label = data[['value']]
            del data['value']
        elif flag == 5:
            gender_mapping = {'Male': 1, 'Female': 0}
            df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y')
            df['New Start date']= df['Start date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            df['End date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y')
            df['New End date'] = df['Start date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['Traveler_gender'] = df['Traveler_gender'].replace(gender_mapping)
            data = df.copy()
            label = data[['Traveler_gender']]
            del data['Traveler_gender']
        elif flag == 6:
            label = data[['Fare']]
            del data['Fare']
        elif flag == 7:
            gender_mapping = {'Male': 1, 'Female': 0}
            df['Gender'] = df['Gender'].replace(gender_mapping)
            data = df.copy()
            label = data[['Total_Amount']]
            del data['Total_Amount']
        elif flag == 8: 
            label = data[['Price']]
            del data['Price']
        elif flag == 9:
            label = data[['lag_price']]
            del data['lag_price']
        elif flag == 10: #yellowtaxi 
            data = df.copy()
            label = data[['fare_amount']]
            del data['fare_amount']
        elif flag == 11: #rideschicago  
            data = df.copy()
            label = data[['fare']]
            del data['fare']
        elif flag == 12:
            label = data[['Rented_Bike_Count']]
            del data['Rented_Bike_Count']
        elif flag == 13:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['weather'] = le.fit_transform(df['weather'])
            data = df.copy()
            label = data[['weather']]
            del data['weather'] 
        elif flag == 14:
            df['HumidityAvgPercent'] = pd.to_numeric(df['HumidityAvgPercent'], errors='coerce')
            df = df.dropna(subset=['HumidityAvgPercent'])
            df['HumidityAvgPercent'] = df['HumidityAvgPercent'].astype(int)
            data = df.copy()
            label = data[['HumidityAvgPercent']]
            del data['HumidityAvgPercent']
        elif flag == 15:
            label = data[['precipitation']]
            del data['precipitation']
        elif flag == 16:
            label = data[['Price']]
            del data['Price']
        elif flag == 17:
            gender_mapping = {'Male': 1, 'Female': 0}
            df['Gender'] = df['Gender'].replace(gender_mapping)
            Customer_mapping = {'Member': 1, 'Normal': 0} 
            df['Customer_type'] = df['Customer_type'].replace(gender_mapping)
            data = df.copy()
            label = data[['Total']]
            del data['Total']
        elif flag == 18:
            gender_mapping = {'Male': 1, 'Female': 0}
            df['Gender'] = df['Gender'].replace(gender_mapping)
            data = df.copy()
            label = data[['LeaveOrNot']]
            del data['LeaveOrNot']

        df_old = df.copy()
        del df_old[label.columns[0]] 
        del df_old['TimeFE_ID']
        print("Original DataFrame shape:", df_old.shape)

        df_time = TimeRelatedFe(data) 
        print("DataFrame after TimeRelatedFe shape:", df_time.shape)
        df_time.to_csv('cleaned_data_flag_{}.csv'.format(flag), index=False)

        df_old = clean_dataframe(df_old)
        print(df_old.columns.tolist())
        
        df_time = clean_dataframe(df_time)

        common_attributes, unique_attributes_df1, unique_attributes_df2 = compare_dataframes(df_old, df_time)
        print("Common attributes:", common_attributes)
        print("Attributes unique to df1:", unique_attributes_df1)
        print("Attributes unique to df2:", unique_attributes_df2)
        
        task_type = 'classification' if label.nunique().values[0] == 2 else 'regression' 

        params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'random_state': 1, 'verbose': 1}
        df_old_save, df_time_save = df_old.copy(), df_time.copy() # new
        score_old, score_time = cross_validated_scores(df_old.copy(), df_time, label, n_jobs, params, task_type)
        
        feature_importances_old_cv = cross_validated_feature_importances(df_old.copy(), label, n_jobs, params, task_type)
        feature_importances_time_cv = cross_validated_feature_importances(df_time.copy(), label, n_jobs, params, task_type)

        print("Cross-validated feature importances before feature generation:\n", feature_importances_old_cv.sort_values(by='importance', ascending=False))
        print("Cross-validated feature importances after feature generation:\n", feature_importances_time_cv.sort_values(by='importance', ascending=False))
        
        save_feature_importances_to_file(feature_importances_old_cv, score_old, task_type, 'Data/Importances/feature_importances_old_flag_{}.txt'.format(flag))
        save_feature_importances_to_file(feature_importances_time_cv, score_time, task_type, 'Data/Importances/feature_importances_new_flag_{}.txt'.format(flag))
        
        if task_type == 'regression':
            print("The RMSE before feature generation is", score_old)
            print("-----------------------------------------")
            print("The RMSE after feature generation is", score_time)
        elif task_type == 'classification':
            print("The ROC AUC before feature generation is", score_old)
            print("-----------------------------------------")
            print("The ROC AUC after feature generation is", score_time)
        elif task_type == 'RMSLE':
            print("The RMSLE before feature generation is", score_old)
            print("-----------------------------------------")
            print("The RMSLE after feature generation is", score_time)
        print("-----------------------------------------")

        print(df_old.columns.tolist())
        pruned_new_df, kept_features = prune(df_old.copy(), df_time, feature_importances_time_cv, 15)
        if len(df) != len(df_time):
            cols = 'different'
            print("Number of rows is different!!")
        else:
            cols = 'same'
            print("Number of rows is the same (:")
        #print(df.columns.tolist())
        pruned_new_df['TimeFE_ID'] = range(1, len(pruned_new_df) + 1)
        df_time['TimeFE_ID'] = range(1, len(df_time) + 1)
        print("#################")
        pruned_new_df = pd.merge(pruned_new_df.copy(), df[['TimeFE_ID',  label.columns[0]]], on='TimeFE_ID', how='left', suffixes=('_new', '_old')) #was df_old not df
        df_time = pd.merge(df_time, df[['TimeFE_ID',  label.columns[0]]], on='TimeFE_ID', how='left', suffixes=('_new', '_old'))
        for col in df_time.columns:
            if col.endswith('_new'):
                original_col = col[:-4]
                old_col = f"{original_col}_old"
                if old_col in df_time.columns:
                    # If columns are identical, drop the old one and rename the new one
                    if df_time[col].equals(df_time[old_col]):
                        df_time.drop(columns=[old_col], inplace=True)
                        df_time.rename(columns={col: original_col}, inplace=True)
                    else:
                        # If columns are not identical, keep both
                        df_time.rename(columns={col: original_col}, inplace=True)
        pruned_df = pruned_new_df.copy()
        feature_importances_pruned_cv = cross_validated_feature_importances(pruned_new_df, label, n_jobs, params, task_type)
        score_prune, score_prune = cross_validated_scores(pruned_new_df, pruned_new_df, label, n_jobs, params, task_type)
        save_feature_importances_to_file(feature_importances_pruned_cv, score_prune, task_type, 'Data/Importances/feature_importances_pruned_flag_{}.txt'.format(flag))
        
        print("Kept Features:")
        print(kept_features)
        df_time.to_csv('Data/Importances/'+cols+'_'+ 'new_df_{}.csv'.format(flag), index=False) 
        pruned_df.to_csv('Data/Importances/pruned_df_{}.csv'.format(flag), index=False) 
