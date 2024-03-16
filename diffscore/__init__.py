import similarity
# temp use registry from similarity
from similarity import make, register


class Env:
    def __new__(cls, env_id: str, *args, **kwarg):
        return similarity.make(f"env.diffscore.{env_id}", *args, **kwargs)


class Measure:
    def __new__(cls, measure_id: str, *args, **kwargs) -> similarity.MeasureInterface:
        # TODO: temp
        if "diffscore" in measure_id:
            print("Deprecation warning: don't include 'diffscore' in measure_id")
            measure_id = measure_id.replace("diffscore.", "")

        return similarity.make(f"measure.diffscore.{measure_id}", *args, **kwargs)


class Card:
    def __new__(cls, card_id: str) -> dict:
        if similarity.is_registered(f"card.{card_id}"):
            return similarity.make(f"card.{card_id}")
        else:
            # TODO: default card?
            print(f"Card {card_id} not found in similarity repository")
            return {"props": []}


class Dataset:
    def __new__(cls, dataset_id: str, *args, **kwargs):
        return similarity.make(f"dataset.diffscore.{dataset_id}", *args, **kwargs)


# important to place imports after class definitions because use the classes in the imports 
from diffscore import analysis
# TODO: rename model to nn?
from diffscore import model
from diffscore import dataset
from diffscore import env
# TODO: have to import training after env because use env in training?
from diffscore import training


