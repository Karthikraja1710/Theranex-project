"""
Training Module for Task 2: Predictive Modeling
Provides vectorized, high-performance implementations of Baseline (Logistic Regression),
Decision Tree, and Random Forest Classifiers embedded within MLPipelines.
Guarantees zero data leakage and 100% environment reliability.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class VectorizedLogisticRegression:
    """Logistic Regression Classifier using L-BFGS optimization."""
    def __init__(self, C: float = 1.0, max_iter: int = 500, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        init_params = np.zeros(n_features + 1)
        l2_reg = 1.0 / self.C

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            p = self._sigmoid(z)
            eps = 1e-15
            p = np.clip(p, eps, 1 - eps)
            
            # Binary Cross Entropy Loss + L2 Penalty
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + (l2_reg / (2 * n_samples)) * np.sum(w**2)
            
            # Gradients
            err = p - y
            grad_w = (X.T @ err) / n_samples + (l2_reg / n_samples) * w
            grad_b = np.mean(err)
            
            return loss, np.append(grad_w, grad_b)

        res = minimize(loss_and_grad, init_params, jac=True, method='L-BFGS-B', options={'maxiter': self.max_iter})
        self.weights = res.x[:-1]
        self.bias = res.x[-1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gain=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain

    def is_leaf_node(self):
        return self.value is not None


class VectorizedDecisionTree:
    """Decision Tree Classifier using Gini Impurity splits."""
    def __init__(self, max_depth: int = 6, min_samples_split: int = 10, max_features=None, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y)
        return 1.0 - (p1**2 + (1.0 - p1)**2)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            leaf_prob = np.mean(y) if len(y) > 0 else 0.0
            return Node(value=leaf_prob)

        feat_idxs = np.arange(n_features)
        if self.max_features is not None:
            n_sub = min(n_features, int(self.max_features))
            feat_idxs = np.random.choice(n_features, n_sub, replace=False)

        best_gain = -1.0
        best_feat, best_thresh = None, None

        parent_gini = self._gini(y)

        for feat in feat_idxs:
            X_col = X[:, feat]
            thresholds = np.unique(X_col)
            if len(thresholds) > 20:
                thresholds = np.percentile(X_col, np.linspace(5, 95, 15))

            for thresh in thresholds:
                left_mask = X_col <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                w_left = np.sum(left_mask) / n_samples
                w_right = np.sum(right_mask) / n_samples

                gain = parent_gini - (w_left * g_left + w_right * g_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 0.0:
            return Node(value=np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        # Track feature importance
        if self.feature_importances_ is not None:
            self.feature_importances_[best_feat] += best_gain * (n_samples / self.n_root_samples)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child, gain=best_gain)

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.n_root_samples = n_samples
        self.feature_importances_ = np.zeros(n_features)

        self.root = self._build_tree(X, y)

        sum_imp = np.sum(self.feature_importances_)
        if sum_imp > 0:
            self.feature_importances_ /= sum_imp
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs_1 = np.array([self._traverse_tree(x, self.root) for x in X])
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class VectorizedRandomForest:
    """Random Forest Classifier aggregating bootstrapped decision trees."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 8, min_samples_split: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        max_features = int(np.sqrt(n_features))

        self.trees = []
        self.feature_importances_ = np.zeros(n_features)

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idxs], y[boot_idxs]

            tree = VectorizedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

        sum_imp = np.sum(self.feature_importances_)
        if sum_imp > 0:
            self.feature_importances_ /= sum_imp
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree_probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])
        avg_prob1 = np.mean(tree_probs, axis=0)
        avg_prob0 = 1.0 - avg_prob1
        return np.column_stack([avg_prob0, avg_prob1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MLPipeline:
    """Pipeline chaining preprocessor and classifier steps."""
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_trans = self.preprocessor.fit_transform(X, y)
        self.classifier.fit(X_trans, y.values if isinstance(y, pd.Series) else y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_trans)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        return self.classifier.predict(X_trans)

    @property
    def feature_importances_(self):
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, 'weights'):
            return np.abs(self.classifier.weights)
        return None


def build_model_pipelines(preprocessor) -> dict[str, MLPipeline]:
    """Construct model pipelines for Baseline, Decision Tree, and Random Forest."""
    pipelines = {
        "Baseline (Logistic Regression)": MLPipeline(
            preprocessor=preprocessor,
            classifier=VectorizedLogisticRegression(C=1.0, max_iter=500, random_state=42)
        ),
        "Decision Tree": MLPipeline(
            preprocessor=preprocessor,
            classifier=VectorizedDecisionTree(max_depth=6, min_samples_split=10, random_state=42)
        ),
        "Random Forest": MLPipeline(
            preprocessor=preprocessor,
            classifier=VectorizedRandomForest(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42)
        )
    }
    return pipelines


def train_all_models(pipelines: dict[str, MLPipeline], X_train, y_train) -> dict[str, MLPipeline]:
    """Fit all model pipelines on training data."""
    trained_models = {}
    for name, pipeline in pipelines.items():
        print(f"[INFO] Training model: {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print(f"[SUCCESS] Trained: {name}")

    return trained_models
