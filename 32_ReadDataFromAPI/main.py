from os import system
import numpy as np
import requests
import json
import pandas as pd
def add_more_features(df):
  df['avg_rooms_per_house'] = df['total_rooms'] / df['households'] #expect positive correlation
  df['avg_persons_per_room'] = df['population'] / df['total_rooms'] #expect negative correlation
  return df

 # Create pandas input function
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = add_more_features(df),
    y = df['median_house_value'] / 100000, # will talk about why later in the course
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

  # Define your feature columns
def create_feature_cols():
  return [
    tf.feature_column.numeric_column('housing_median_age'),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), boundaries = np.arange(32.0, 42, 1).tolist()),
    tf.feature_column.numeric_column('avg_rooms_per_house'),
    tf.feature_column.numeric_column('avg_persons_per_room'),
    tf.feature_column.numeric_column('median_income')
  ]
# Create estimator train and evaluate function
def train_and_evaluate(output_dir, num_train_steps):
  estimator = tf.estimator.LinearRegressor(model_dir = output_dir, feature_columns = create_feature_cols())
  train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(traindf, None), 
                                      max_steps = num_train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn(evaldf, 1), 
                                    steps = None, 
                                    start_delay_secs = 1, # start evaluating after N seconds, 
                                    throttle_secs = 5)  # evaluate every N seconds
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
 
system('cls')

dateFrom='20190101'
dateUpto='20190110'
sysCode='IMT'
getUrl='https://eeuat.ebixcash.com/ebixcashweb/webapi/api/SalesSummaryData/'+sysCode+'/'+dateFrom+'/'+dateUpto


response = requests.get(getUrl, auth=('apiusername', 'securepassword'))
jsonData = json.loads(response.text)
df = pd.io.json.json_normalize(jsonData)

df.transactionDate = pd.to_datetime(df.transactionDate)
df['dayOfWeek'] = df.transactionDate.dt.day_name()

df=df.drop(df[df.principleName=='VIA'].index)
df=df.drop(['transactionCount'],axis=1)
#df = pd.get_dummies(df['time'].dt.time.map(a))
print(df.principleName.unique())
print(df.head())
print(df.describe())

np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]

print(traindf.head())
print(evaldf.describe())

train_and_evaluate(OUTDIR, 2000)