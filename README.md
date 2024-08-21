# Bachelor_Thesis
Code for the Bachelor Thesis of Alexis Niermann
To apply TimeFE to your dataset, run your training data through "df_time = TimeRelatedFe(data)" in TimeRelatedFE.py

If you want to prune it, run 
data['TimeFE_ID'] = range(1, len(data) + 1)
df_time = TimeRelatedFe(data)
df_time = clean_dataframe(df_time)
feature_importances_time_cv = cross_validated_feature_importances(df_time.copy(), label, n_jobs, params, task_type)

(set features_I_want_to_keep to how many top n generated features you want to keep)

pruned_new_df, kept_features = prune(df_old.copy(), df_time, feature_importances_time_cv, features_I_want_to_keep)
