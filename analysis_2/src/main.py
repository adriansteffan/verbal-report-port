from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

from extractors.random import RandomFeatureExtractor
from pipeline import evaluate

### Chance baseline, no LLM calls

evaluate(make_pipeline(RandomFeatureExtractor(), DummyClassifier(strategy="uniform")))


### The taxonomy extractor. Four independent choices, each option carrying only
### the settings that apply to it. Every LLM call is cached in
### output/llm_cache.sqlite, so re-runs are free and only new (prompt, seed)
### pairs cost time. Scout with limit= and n_seeds=1 before paying for a config.

# from sklearn.ensemble import RandomForestClassifier
#
# from extractors.taxonomy.levels import Categories, Clusters
# from extractors.taxonomy.coders import Binary, TopK
# from extractors.taxonomy.extractor import TaxonomyExtractor
# from extractors.taxonomy.segments import Groups, Transcript, Utterances
# from extractors.taxonomy.scopes import Acquisition, UntilDiscovery
#
# evaluate(
#     make_pipeline(
#         TaxonomyExtractor(
#             scope=Acquisition(),
#             segments=Transcript(),
#             coder=TopK(k=3, n_seeds=1),
#             level=Categories(),
#         ),
#         RandomForestClassifier(random_state=0),
#     ),
#     limit=10,
# )


### Sweeping: nested options expose sklearn param paths, e.g. coder__k

# MODELS = ["qwen3.6:27b", "qwen3.5:35b-a3b", "gemma4:31b"]  # chat models on the endpoint
#
# for model in MODELS:
#     for level in [Categories(), Clusters(across="max")]:
#         for coder in [TopK(k=1, model=model), TopK(k=3, model=model), Binary(model=model)]:
#             evaluate(
#                 make_pipeline(
#                     TaxonomyExtractor(
#                         scope=UntilDiscovery(unaware_extra_rounds=12),
#                         segments=Groups(size=6, pooling="max"),
#                         coder=coder,
#                         level=level,
#                     ),
#                     RandomForestClassifier(random_state=0),
#                 )
#             )
