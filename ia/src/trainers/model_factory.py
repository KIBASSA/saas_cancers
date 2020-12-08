from simple_trainer import ModelTrainer, SimpleClassifierModelTrainer
from enum import Enum
class ModelTrainerType(Enum):
    simple_classifier = 1
    gan_classifier = 2
    gan_distributed_classifier = 3

class ModelTrainerFactory(object):
    def get_trainer(self, model_trainer_type:ModelTrainerType):
        if model_trainer_type.value == ModelTrainerType.simple_classifier.value:
            return SimpleClassifierModelTrainer()
        elif model_trainer_type.value == ModelTrainerType.gan_classifier.value:
            return ModelTrainer()