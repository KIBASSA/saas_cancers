
class AbstractModelTrainer(object):
    def __init__(self):
        self.IMAGE_RESIZE = 50
        #self.BATCH_SIZE_TRAINING_LABELED_SUBSET = 16
        self.BATCH_SIZE_TRAINING_LABELED_SUBSET = 256
        self.BATCH_SIZE_TRAINING_UNLABELED_SUBSET = 8
    
    def set_params(self, run, epochs):
        self.run = run
        self.epochs = epochs



