"""
Duolingo SLAM Shared Task - Baseline Model

This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the eval.py script.

"""

import argparse
from io import open
import os
import sys
import numpy as np

from dataio import load_data, DataSource, save_features_to_file
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from future.utils import iteritems


def main():
    """
    This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred', required=True)
    parser.add_argument('--task', help=' Output file name for task, defaults to test_name.pred')
    parser.add_argument('--max-train', help='max train instances', type=int,default=None)
    parser.add_argument('--max-test', help='max test instances', type=int,default=None)
    parser.add_argument('--show-top-features', help='show top-20 features', default=False, action="store_true")
    parser.add_argument('--learner', help='select ML', choices=("logReg", "svm", "sgd"), default="logReg")
    parser.add_argument('--C', help='regularization parameter C, smaller values specify stronger regularization', type=float,default=1.0)
    parser.add_argument('--cv', help='use stratified k fold ("scrambled")', default=None, type=int)
    parser.add_argument('--lang', help="language pair: en_es es_en fr_en", required=True)
    parser.add_argument('--feats', help="active feature blocks", default="pos")
    parser.add_argument('--save_features_to_file', default=False,action='store_true')
    parser.add_argument('--feature_degree', default=1,type=int,choices=[1,2])
    parser.add_argument('--class-weight', default=None,choices=("None","balanced"))
    parser.add_argument('--penalty', default='l2',choices=("none", "l2", "l1","elasticnet"))
    parser.add_argument('--solver', default='liblinear',choices=("newton-cg","lbfgs","liblinear","sag","saga"))
    parser.add_argument('--format_wise_models', default=False,action="store_true")

    ## SGD parameter
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--loss', default='log')


    args = parser.parse_args()

    if not args.pred:
        args.pred = args.test + '.pred'
    if not args.pred:
        args.task = args.test + '.task'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    if args.class_weight == "None":
        args.class_weight = None

    # Assert that the train course matches the test course
    #assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    data_source = DataSource(args.lang)

    training_data, training_labels = load_data(args.train, cv=args.cv, train=True,data_source=data_source)
    test_data, _ = load_data(args.test, train=False ,data_source=data_source)

    active_features = args.feats.split(":") if ":" in args else args.feats

    ####################################################################################
    # Here is the delineation between loading the data and running the baseline model. #
    # Replace the code between this and the next comment block with your own.          #
    ####################################################################################

    if args.cv:
        all_data = np.array(training_data + test_data)
        # overwrite to have them for whole set
        training_labels = [instance_data.label for instance_data in all_data]

        user_ids = [instance_data.instance_id for instance_data in all_data]

        skf = StratifiedKFold(n_splits=2)
        fold_num=0
        for train_index, test_index in skf.split(all_data, user_ids):
            print("fold {}".format(fold_num))
            
            training_data = all_data[train_index]
            test_data = all_data[test_index]
            # if max_train is None, training_data == training_data[:], so no need for if-else
            X_train = [instance_data.to_features(data_source, active_features) for instance_data in training_data][:args.max_train]
            Y_train = [training_labels[instance_data.instance_id] for instance_data in training_data][:args.max_train]

            X_test = [instance_data.to_features(data_source, active_features) for instance_data in test_data][:args.max_test]
            ids_test = [instance_data.instance_id for instance_data in test_data][:args.max_test]

            train_eval(args, X_train, Y_train, X_test, ids_test, output_filename=args.pred + ".fold{}".format(fold_num), evaluate=True)
            fold_num+=1
    else:
        # if max_train is None, training_data == training_data[:], so no need for if-else
        id_2_format = dict()
        formats = set()
        models_for_each_format = dict()
        for instance_data in training_data:
            id_2_format[instance_data.instance_id]=instance_data.format
            formats.add(instance_data.format)
        for instance_data in test_data:
            id_2_format[instance_data.instance_id]=instance_data.format
            formats.add(instance_data.format)
        X_train = [instance_data.to_features(data_source, active_features) for instance_data in training_data][:args.max_train]
        Y_train = [training_labels[instance_data.instance_id] for instance_data in training_data][:args.max_train]
        ids_train = [instance_data.instance_id for instance_data in training_data][:args.max_train]

        X_test = [instance_data.to_features(data_source, active_features) for instance_data in test_data][:args.max_test]
        ids_test = [instance_data.instance_id for instance_data in test_data][:args.max_test]

        if args.format_wise_models:
            total_preds = []
            total_ids = []
            for current_format in formats:
                # TODO Barbara here is where I declare the model for each task format
                models_for_each_format[current_format] = get_learner(args)
                X_train_current = [instance_data.to_features(data_source, active_features) for instance_data in training_data if instance_data.format == current_format][
                          :args.max_train]
                Y_train_current = [training_labels[instance_data.instance_id] for instance_data in training_data if instance_data.format == current_format][:args.max_train]
                X_test_current = [instance_data.to_features(data_source, active_features) for instance_data in test_data if instance_data.format == current_format][
                         :args.max_test]
                ids_test_current = [instance_data.instance_id for instance_data in test_data if instance_data.format == current_format ][:args.max_test]
                vectorizer = DictVectorizer()
                X_train_current = vectorizer.fit_transform(X_train_current)
                X_test_current = vectorizer.transform(X_test_current)
                clf = models_for_each_format[current_format]
                clf.fit(X_train_current,Y_train_current)
                pred_scores = [x[1] for x in models_for_each_format[current_format].predict_proba(X_test_current)]
                total_preds.extend(pred_scores)
                total_ids.extend(ids_test_current)

                if args.show_top_features:
                    print("features for format: {}".format(current_format))
                    show_most_informative_features(clf, vectorizer)

            predictions = dict([(instance_id, pred_score) for instance_id, pred_score in zip(total_ids, total_preds)])
            with open(args.pred, 'wt') as f:
                for instance_id, prediction in iteritems(predictions):
                    f.write(instance_id + ' ' + str(prediction) + '\n')
        else:
            train_eval(args, X_train, Y_train, X_test, ids_test, output_filename=args.pred)

def get_learner(args):
    ## weight='auto' ; L2 + l2 / elastinet --> change to log-scaled weighting? (weights) [=> penalty - solver] 
    clf = LogisticRegression(C=args.C, class_weight=args.class_weight, penalty=args.penalty, solver=args.solver)
    if args.learner == "svm":
        clf = SVC(C=args.C)
    if args.learner == "sgd":
        clf = SGDClassifier(class_weight=args.class_weight, penalty=args.penalty, loss=args.loss, tol=1e-1, max_iter=args.max_iter)
    return clf

def train_eval(args, X_train, Y_train, X_test, ids_test, output_filename=None, evaluate=False):
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    print("#vocab: {}".format(len(vectorizer.vocabulary_)))
    print("features: {}".format(args.feats))
    if args.feature_degree == 2:
        poly = PolynomialFeatures(degree=args.feature_degree, include_bias=False)
        X_train = poly.fit_transform(X_train.toarray())
    sys.stderr.write("size of training data" + str(X_train.shape) + "\n")

    clf = get_learner(args)
    clf.fit(X_train, Y_train)

    sys.stderr.write(str(clf))

    X_test = vectorizer.transform(X_test)
    if args.feature_degree == 2:
        X_test = poly.transform(X_test.toarray())
    sys.stderr.write("size of test data" + str(X_test.shape) + "\n")
    preds_scores = [x[1] for x in clf.predict_proba(X_test)]
    predictions = dict([(instance_id, pred_score) for instance_id, pred_score in zip(ids_test, preds_scores)])

    if args.show_top_features:
        show_most_informative_features(clf, vectorizer)

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    with open(args.pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


def show_most_informative_features(classifier, vectorizer, n=20):
    """
    output features with the highest weights
    :param n: top-n
    """
    feature_names = vectorizer.get_feature_names()
    for i in range(0,len(classifier.coef_)):
        coefs_with_fns = sorted(zip(classifier.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        print("Coefficients for class: {}".format(classifier.classes_[i]))
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))



if __name__ == '__main__':
    main()
