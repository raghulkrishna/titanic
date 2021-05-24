import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import os
import dvc.api

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

repo='/Users/raghulkrishna/Desktop/projects/mlops/titanic'
train_resource_url = dvc.api.get_url(
    path='train.csv',
    repo=repo,
    rev='v2'

    )
eval_resource_url = dvc.api.get_url(
    path='eval.csv',
    repo=repo,
    rev='v2'

    )

mlflow.set_tracking_uri(os.environ['MLFLOWURI'])
mlflow.tensorflow.autolog()
mlflow.set_experiment("tensorflow experiment")

with mlflow.start_run():
    mlflow.log_param('train_url',train_resource_url)
    mlflow.log_param('eval_url',eval_resource_url)


    # Load dataset.
    dftrain = pd.read_csv(train_resource_url)
    dfeval = pd.read_csv(eval_resource_url)
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')
    mlflow.log_param('train_shape',str(dftrain.shape))
    mlflow.log_param('eval_shape',str(dfeval.shape))


    fc = tf.feature_column
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
      
    def one_hot_cat_column(feature_name, vocab):
        return tf.feature_column.indicator_column(
          tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                                    vocab))
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        
      # Need to one-hot encode categorical features.
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
        

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                              dtype=tf.float32))

    example = dict(dftrain.head(1))
    class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
    print('Feature value: "{}"'.format(example['class'].iloc[0]))
    print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())


    # Use entire batch since this is such a small dataset.
    NUM_EXAMPLES = len(y_train)

    def make_input_fn(X, y, n_epochs=None, shuffle=True):
        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(NUM_EXAMPLES)
            # For training, cycle thru dataset as many times as need (n_epochs=None).    
            dataset = dataset.repeat(n_epochs)
            # In memory training doesn't use batching.
            dataset = dataset.batch(NUM_EXAMPLES)
            return dataset
        return input_fn

    # Training and evaluation input functions.
    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

    params = {
      'n_trees': 50,
      'max_depth': 3,
      'n_batches_per_layer': 1,
      # You must enable center_bias = True to get DFCs. This will force the model to 
      # make an initial prediction before using any features (e.g. use the mean of 
      # the training labels for regression or log odds for classification when
      # using cross entropy loss).
      'center_bias': True

    }
    f_col = {"features":list(dftrain.columns)}

    mlflow.log_dict(f_col,"feature columns.json")


    est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
    # Train model.
    est.train(train_input_fn, max_steps=100)

    # Evaluation.
    results = est.evaluate(eval_input_fn)

    # Make predictions.
    pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))
    df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
    # Plot results.
    ID = 182
    example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.
    TOP_N = 8  # View top 8 features.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index
    ax = example[sorted_ix].plot(kind='barh')

    importances = est.experimental_feature_importances(normalize=True)
    df_imp = pd.DataFrame(pd.Series(importances)).reset_index()
    df_imp.columns=["feature","importance"]
    # Visualize importances.
    N = 8
    ax = df_imp.iloc[0:N][::-1].plot(kind='barh')
    ax.figure.savefig("feature.png")
    axis_fs = 18 #fontsize
    title_fs = 22 #fontsize
    sns.set(style="whitegrid")
    mlflow.log_figure(ax.figure, "feature.png")



    ax = sns.barplot(x="importance", y="feature", data=df_imp)
    ax.set_xlabel('Importance',fontsize = axis_fs) 
    ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
    ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

    plt.tight_layout()
    plt.savefig("feature_importance.png",dpi=120) 
    plt.close()
    mlflow.log_figure(ax.figure, "feature_importance.png")


