from pathlib import Path
from functools import partial

from diffscore import Measure, Card
from diffscore.analysis import pipeline_optim_score, decoder_logistic


datasets = [
    "ultrametric",
    "MajajHong2015",
    "FreemanZiemba2013",
    "Hatsopoulos2007"
]
measures = [
    "procrustes-angular-score", 
    "cka"
]
# all the scoring measures
# measures = [
#     measure_id.split(".")[-1] for measure_id in Measure("*").keys() if "score" in Card(measure_id.split(".")[-1])["props"]
# ]
# TODO: cca-angular-score MajajHong NaN grad
print(measures)


# save score by measure and aggregate everything in another function
for dataset in datasets:
    pipeline_optim_score(
        dataset=dataset,
        measure=measures,
        stop_score=0.99,
        decoder="logistic",
        save_dir=Path(__file__).parent / "results" / dataset,
    )