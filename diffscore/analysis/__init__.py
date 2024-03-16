
import similarity


# class Measure:
#     def __new__(cls, measure_id: str, *args, **kwargs) -> similarity.MeasureInterface:
#         # TODO: temp
#         if "diffscore" in measure_id:
#             print("Deprecation warning: don't include 'diffscore' in measure_id")
#             measure_id = measure_id.replace("diffscore.", "")

#         return similarity.make(f"measure.diffscore.{measure_id}", *args, **kwargs)


# class Card:
#     def __new__(cls, card_id: str) -> dict:
#         if similarity.is_registered(f"card.{card_id}"):
#             return similarity.make(f"card.{card_id}")
#         else:
#             # TODO: default card?
#             print(f"Card {card_id} not found in similarity repository")
#             return {"props": []}


# class Dataset:
#     def __new__(cls, dataset_id: str) -> similarity.DatasetInterface:
#         return similarity.make(f"dataset.diffscore.{dataset_id}")


from . import measures
from .captured_pcs import pc_captured_variance, pipeline_optim_score
from .decoding import decoder_logistic
