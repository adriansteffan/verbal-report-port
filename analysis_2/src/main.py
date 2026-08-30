from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline
from extractors import base
# from pipeline import evaluate

pipeline.PROGRESS = True
# MODEL = "phi4-mini:latest"
MODEL = "qwen3.8:27b"
base.WORKERS = 1
EXPORT_CALLS = True

CLASSIFIERS = [
    # DummyClassifier(strategy="uniform"),
    make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
    # make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced")),
    # LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    # GaussianNB(),
    # RandomForestClassifier(
    #   n_estimators=100,
    #   max_depth=3,
    #   min_samples_leaf=5,
    #   random_state=0,
    #   class_weight="balanced",
    # ),
]


# evaluate(make_pipeline(RandomFeatureExtractor(), DummyClassifier(strategy="uniform")))


from extractors.taxonomy.coders import TopK
from extractors.taxonomy.extractor import TaxonomyExtractor
from extractors.taxonomy.levels import Categories, Clusters
from extractors.taxonomy.scopes import Acquisition
from extractors.taxonomy.segments import Groups
from pipeline import export_calls, features, results_table, unit_features

SWEEP = [
    TaxonomyExtractor(
        Acquisition(),
        Groups(size=6, pooling=pooling),
        TopK(k=3, n_seeds=5, memory=False, shuffle="clustered", model=MODEL),
        level,
    )
    for pooling in ["max", "mean"]
    for level in [Categories(), Clusters(across="max"), Clusters(across="mean")]
]


for extractor in SWEEP:
    features(extractor)
    unit_features(extractor)
    if EXPORT_CALLS:
        export_calls(extractor)
#    for classifier in CLASSIFIERS:
#        evaluate(make_pipeline(extractor, classifier), n_permutations=500)

# results_table()
