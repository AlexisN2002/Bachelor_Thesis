# Bachelor_Thesis
Code for the Bachelor Thesis of Alexis Niermann

#TimeRelatedFE.py

Explanation: TimeRelatedFE.py applies various time-related transformations to your dataset to extract hidden temporal information in the features, giving your model better time context while retaining your original features. Due to the use of LightGBM, the features may be renamed.

Usage: To use this script, go to the main function and add your file paths to the your_csv_files = [] list. Then, uncomment the line <for flag, csv_file_path in enumerate(your_csv_files, start=1):> and remove the line below it (i.e., the old for loop). Additionally, fill out the "your flags here" section like the template for all your files in the correct order, and remove the existing if-elif block. Make sure to update the file paths where the results will be saved.

TimeRelatedFE.py will transform all your data and provide you with the unpruned new dataset, as well as a pruned version that includes the top 15 new features. Both versions will be indicated by their name and position in your CSV file list. Additionally, three .txt files containing the scores and importances of the original, unpruned, and pruned datasets will be generated. The task type will be determined automatically (using roc_auc for classification and rmse for regression).

#Baseline_Calculation.py

Explanation: In this script, your newly generated (unpruned) DataFrame will be further enhanced using OpenFE to potentially create a more robust dataset. For comparison, you will see how well OpenFE on your original dataset would have performed.

Usage: Edit the main function as before by inserting your list of CSV files and their corresponding label columns. Also, update the run_experiment function with your own CSV files and labels. You will be provided with the datasets and .txt files containing the scores and importances. Be sure to update the file paths if necessary.








//To apply TimeFE to your dataset, run your training data through "df_time = TimeRelatedFe(data)" in TimeRelatedFE.py

//If you want to prune it, run 
//data['TimeFE_ID'] = range(1, len(data) + 1)
//df_time = TimeRelatedFe(data)
//df_time = clean_dataframe(df_time)
//feature_importances_time_cv = cross_validated_feature_importances(df_time.copy(), label, n_jobs, params, task_type)

//(set features_I_want_to_keep to how many top n generated features you want to keep)

//pruned_new_df, kept_features = prune(df_old.copy(), df_time, feature_importances_time_cv, features_I_want_to_keep)
