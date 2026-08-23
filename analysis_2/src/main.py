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
    # GaussianNB assumes the features are conditionally independent, but top-k
    # forces exactly k categories per unit, so they are compositional and
    # negatively coupled by construction
    # GaussianNB(),
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
from extractors.taxonomy.segments import Groups, Transcript, Utterances
from pipeline import features


# for memory in [False, True]:
#     for scope in [Acquisition(), UntilDiscovery()]:
#         for segments in [Transcript(), Groups(size=6)]:
#             for level in [Categories(), Clusters()]:
#                 features(
#                     TaxonomyExtractor(
#                         scope, segments, TopK(k=3, n_seeds=5, memory=memory), level
#                     )
#                 )


# for level in [Categories(), Clusters()]:
#     features(
#         TaxonomyExtractor(Acquisition(), Utterances(), TopK(k=3, n_seeds=1), level)
#     )


SWEEP = [
    TaxonomyExtractor(scope, segments, TopK(k=3, n_seeds=5, memory=memory), level)
    for scope in [Acquisition(), UntilDiscovery()]
    for level in [Categories(), Clusters(across="max")]
    for segments, memory in [(Transcript(), False)]
    + [
        (Groups(size=6, pooling=pooling), memory)
        for pooling in ["max", "mean"]
        for memory in [False, True]
    ]
] + [
    TaxonomyExtractor(
        Acquisition(), Utterances(pooling=pooling), TopK(k=3, n_seeds=1), level
    )
    for level in [Categories(), Clusters(across="max")]
    for pooling in ["max", "mean"]
]

for extractor in SWEEP:
    features(extractor)
    for classifier in CLASSIFIERS:
        evaluate(make_pipeline(extractor, classifier), n_permutations=500)
