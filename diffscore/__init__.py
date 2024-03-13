import similarity
# temp use registry from similarity
from similarity import make, register

from . import analysis
from . import model
from . import dataset
from . import env
from . import training

from .analysis import Measure
from .env import Env

# from diffscore import analysis as analysis
# # TODO: rename model to nn?
# from diffscore import model as model
# from diffscore import dataset as dataset
# from diffscore import env as env
# # TODO: have to import training after env because use env in training?
# from diffscore import training as training

# from diffscore.analysis import Measure
# from diffscore.env import Env
