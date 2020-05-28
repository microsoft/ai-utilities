import argparse
import os

import lightgbm as lgb
import pandas as pd
from azureml.core import Run
import joblib
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from azure_utils.machine_learning.item_selector import ItemSelector

if __name__ == '__main__':
# """ Main Method to use with AzureML"""
    # Define the arguments.
    parser = argparse.ArgumentParser(description='Fit and evaluate a model based on train-test datasets.')
    parser.add_argument('-d', '--train_data', help='the training dataset name', default='balanced_pairs_train.tsv')
    parser.add_argument('-t', '--test_data', help='the test dataset name', default='balanced_pairs_test.tsv')
    parser.add_argument('-i', '--estimators', help='the number of learner estimators', type=int, default=1)
    parser.add_argument('--min_child_samples', help='the minimum number of samples in a child(leaf)', type=int,
                        default=20)
    parser.add_argument('-v', '--verbose', help='the verbosity of the estimator', type=int, default=-1)
    parser.add_argument('-n', '--ngrams', help='the maximum size of word ngrams', type=int, default=1)
    parser.add_argument('-u', '--unweighted', help='do not use instance weights', action='store_true', default=False)
    parser.add_argument('-m', '--match', help='the maximum number of duplicate matches', type=int, default=20)
    parser.add_argument('--outputs', help='the outputs directory', default='.')
    parser.add_argument('--inputs', help='the inputs directory', default='.')
    parser.add_argument('-s', '--save', help='save the model', action='store_true', default=True)
    parser.add_argument('--model', help='the model file', default='model.pkl')
    parser.add_argument('--instances', help='the instances file', default='inst.txt')
    parser.add_argument('--labels', help='the labels file', default='labels.txt')
    parser.add_argument('-r', '--rank', help='the maximum rank of correct answers', type=int, default=3)
    args = parser.parse_args()

    run = Run.get_context()

    # The training and testing datasets.
    inputs_path = args.inputs
    data_path = os.path.join(inputs_path, args.train_data)
    test_path = os.path.join(inputs_path, args.test_data)

    # Create the outputs folder.
    outputs_path = args.outputs
    os.makedirs(outputs_path, exist_ok=True)
    model_path = os.path.join(outputs_path, args.model)
    instances_path = os.path.join(outputs_path, args.instances)
    labels_path = os.path.join(outputs_path, args.labels)

    # Load the training data
    print('Reading {}'.format(data_path))
    train = pd.read_csv(data_path, sep='\t', encoding='latin1', error_bad_lines=False)

    # Limit the number of duplicate-original question matches.
    train = train[train.n < args.match]

    # Define the roles of the columns in the training data.
    feature_columns = ['Text_x', 'Text_y']
    label_column = 'Label'
    duplicates_id_column = 'Id_x'
    answer_id_column = 'AnswerId_y'

    # Report on the training dataset: the number of rows and the proportion of true matches.
    print('train: {:,} rows with {:.2%} matches'.format(
        train.shape[0], train[label_column].mean()))

    # Compute the instance weights used to correct for class imbalance in training.
    weight_column = 'Weight'
    if args.unweighted:
        weight = pd.Series([1.0], train[label_column].unique())
    else:
        label_counts = train[label_column].value_counts()
        weight = train.shape[0] / (label_counts.shape[0] * label_counts)
    train[weight_column] = train[label_column].apply(lambda x: weight[x])

    # Collect the unique ids that identify each original question's answer.
    labels = sorted(train[answer_id_column].unique())
    label_order = pd.DataFrame({'label': labels})

    # Collect the parts of the training data by role.
    train_x = train[feature_columns]
    train_y = train[label_column]
    sample_weight = train[weight_column]

    # Use the inputs to define the hyperparameters used in training.
    n_estimators = args.estimators
    min_child_samples = args.min_child_samples
    if args.ngrams > 0:
        ngram_range = (1, args.ngrams)
    else:
        ngram_range = None

    # Verify that the hyperparameter values are valid.
    assert n_estimators > 0
    assert min_child_samples > 1
    assert isinstance(ngram_range, tuple) and len(ngram_range) == 2
    assert 0 < ngram_range[0] <= ngram_range[1]

    # Define the pipeline that featurizes the text columns.
    featurization = [
        (column,
         make_pipeline(ItemSelector(column),
                       text.TfidfVectorizer(ngram_range=ngram_range)))
        for column in feature_columns]
    features = FeatureUnion(featurization)

    # Define the estimator that learns how to classify duplicate-original question pairs.
    estimator = lgb.LGBMClassifier(n_estimators=n_estimators,
                                   min_child_samples=min_child_samples,
                                   verbose=args.verbose)

    # Define the model pipeline as feeding the features into the estimator.
    model = Pipeline([
        ('features', features),
        ('model', estimator)
    ])

    # Fit the model.
    print('Training...')
    model.fit(train_x, train_y, model__sample_weight=sample_weight)

    # Save the model to a file, and report on its size.
    if args.save:
        joblib.dump(model, model_path)
        print('{} size: {:.2f} MB'.format(model_path, os.path.getsize(model_path) / (2 ** 20)))
