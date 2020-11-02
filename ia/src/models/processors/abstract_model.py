
class AbstractProcessorModel(object):
    def __init__(self):
        self.IMAGE_RESIZE = 50
        self.BATCH_SIZE_TRAINING_LABELED_SUBSET = 16
        self.BATCH_SIZE_TRAINING_UNLABELED_SUBSET = 8
    
