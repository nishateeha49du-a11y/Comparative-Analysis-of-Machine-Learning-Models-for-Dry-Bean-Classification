from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


SEEDS = [42, 10, 2026]


def get_models(random_state=42):

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state,
            max_iter=5000
        ),

        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "AdaBoost": AdaBoostClassifier(
            random_state=random_state
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=random_state
        ),

        "Multi-layer Perceptron": MLPClassifier(
            max_iter=1000,
            random_state=random_state
        )
    }

    return models