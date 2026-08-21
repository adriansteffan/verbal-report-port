from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pipeline
from extractors.random import RandomFeatureExtractor
from pipeline import evaluate

pipeline.PROGRESS = True  # per-participant bars while extracting

CLASSIFIERS = [
    # DummyClassifier(strategy="uniform"),
    make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
    make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced")),
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    GaussianNB(),
    # RandomForestClassifier(
    #   n_estimators=100,
    #   max_depth=3,
    #   min_samples_leaf=5,
    #   random_state=0,
    #   class_weight="balanced",
    # ),
]


evaluate(make_pipeline(RandomFeatureExtractor(), DummyClassifier(strategy="uniform")))


from extractors.taxonomy.coders import TopK
from extractors.taxonomy.extractor import TaxonomyExtractor
from extractors.taxonomy.levels import Categories, Clusters
from extractors.taxonomy.scopes import Acquisition, UntilDiscovery
from extractors.taxonomy.segments import Groups, Transcript
from pipeline import features


for memory in [False, True]:
    for scope in [Acquisition(), UntilDiscovery()]:
        for segments in [Transcript(), Groups(size=6)]:
            for level in [Categories(), Clusters()]:
                features(
                    TaxonomyExtractor(
                        scope, segments, TopK(k=3, n_seeds=5, memory=memory), level
                    )
                )

# for memory in [False, True]:
#     for scope in [Acquisition(), UntilDiscovery()]:
#         for level in [Categories(), Clusters(across="max")]:
#             # Transcript makes one unit, so it has nothing to pool
#             segmenters = [Transcript()] + [
#                 Groups(size=6, pooling=p) for p in ["max", "mean"]
#             ]
#             for segments in segmenters:
#                 extractor = TaxonomyExtractor(
#                     scope, segments, TopK(k=3, n_seeds=5, memory=memory), level
#                 )
#                 for classifier in CLASSIFIERS:
#                     evaluate(make_pipeline(extractor, classifier), n_permutations=1000)
