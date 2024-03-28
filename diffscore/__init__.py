# use similarity's registration system
import similarity


def make(id, *args, **kwargs):
    if "diffscore" in id:
        raise ValueError(f"Don't include 'diffscore' in id, {id}")
    assert len(id.split('.')) == 2, f"Invalid id {id}, must be of the form 'type.name'"
    # insert backend prefix
    # e.g. measure.cka => measure.diffscore.cka
    id = f"{id.split('.')[0]}.diffscore.{id.split('.')[1]}"
    return similarity.make(id, *args, **kwargs)


def register(id, *args, **kwargs):
    if "diffscore" in id:
        raise ValueError(f"Don't include 'diffscore' in id, {id}")
    assert len(id.split('.')) == 2, f"Invalid id {id}, must be of the form 'type.name'"
    # insert backend prefix
    id = f"{id.split('.')[0]}.diffscore.{id.split('.')[1]}"
    return similarity.register(id, *args, **kwargs)


class Env:
    def __new__(cls, env_id: str, *args, **kwargs):
        if similarity.is_registered(f"env.diffscore.{env_id}"):
            return similarity.make(f"env.diffscore.{env_id}", *args, **kwargs)
        else:
            raise ValueError(f"Env {env_id} not found in repository")


class Measure:
    def __new__(cls, measure_id: str, *args, **kwargs) -> similarity.MeasureInterface:
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
            print(f"Card {card_id} not found in repository")
            return {"props": []}


class Dataset:
    def __new__(cls, dataset_id: str, *args, **kwargs):
        return similarity.make(f"dataset.diffscore.{dataset_id}", *args, **kwargs)


# important to place imports after class definitions because use the classes in the imports
from diffscore import analysis
from diffscore import nn
from diffscore import model  # backward compatibility
from diffscore import dataset
from diffscore import env
# have to import training after env because use env in training
from diffscore import training
