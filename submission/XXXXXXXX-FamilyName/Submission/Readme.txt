The submission is a csv file containing for each entry of the test set
the following columns:
- KEY (which identifies the road)
- KM (the kilometer of the sensor)
- DATETIME_UTC (the timestamp)
- PREDICTION_STEP (to distinguish the prediction quarter)
- SPEED_AVG (the predicted value)

The first row of the csv will be
KEY,KM,DATETIME_UTC,PREDICTION_STEP,SPEED_AVG

Followed by the actual data
