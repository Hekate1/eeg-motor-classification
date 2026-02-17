import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from data_utils import load_multi_subject_data

class RiemannianBaseline:
    """
    Classical Riemannian Tangent-space pipeline with a shallow logistic regression head.
    """
    def __init__(self, estimator='lwf', metric='riemann', clf=None):
        # Covariance estimator and tangent-space mapper
        self.cov = Covariances(estimator=estimator)
        self.ts = TangentSpace(metric=metric)
        # Shallow classifier (logistic regression by default)
        self.clf = clf if clf is not None else LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        # Estimate covariance matrices
        covmats = self.cov.fit_transform(X)
        # Map to tangent space
        features = self.ts.fit_transform(covmats, y)
        # Fit classifier
        self.clf.fit(features, y)
        return self

    def predict(self, X):
        # Estimate covariance matrices
        covmats = self.cov.transform(X)
        # Map to tangent space
        features = self.ts.transform(covmats)
        # Predict labels
        return self.clf.predict(features)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def evaluate_multi_subject_riemannian(train_subjects, test_subjects, run_ids, use_motor_channels):
    """
    Train RiemannianBaseline on train_subjects and evaluate on test_subjects.
    """
    # Load training data
    X_train, y_train, _ = load_multi_subject_data(
        subject_ids=train_subjects,
        run_ids=run_ids,
        use_motor_channels=use_motor_channels
    )
    # Load test data
    X_test, y_test, _ = load_multi_subject_data(
        subject_ids=test_subjects,
        run_ids=run_ids,
        use_motor_channels=use_motor_channels
    )
    # Instantiate and train the baseline
    baseline = RiemannianBaseline()
    baseline.fit(X_train, y_train)
    # Evaluate
    acc = baseline.score(X_test, y_test)
    return acc 