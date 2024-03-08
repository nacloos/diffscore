from . import measures

import similarity


class Measure:
    def __new__(cls, measure_id: str, *args, **kwargs) -> similarity.MeasureInterface:
        # TODO: temp
        if "diffscore" in measure_id:
            print("Deprecation warning: don't include 'diffscore' in measure_id")
            measure_id = measure_id.replace("diffscore.", "")

        return similarity.make(f"measure.diffscore.{measure_id}", *args, **kwargs)
