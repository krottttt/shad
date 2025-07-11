from contest.file import extract_hybrid_strategy_features,read_timeseries,build_datasets,extract_advanced_features,train_models,predict,score_models



train_ts, val_ts = read_timeseries('./demand-forecasting-kernels-only/train.csv')

models = train_models(train_ts,5)
print(models)
predict(val_ts,models,extract_advanced_features)
score_models(train_ts,val_ts,models,predict,extract_advanced_features)