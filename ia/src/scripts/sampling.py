
from azureml.core import Run
from tensorflow.keras.preprocessing import image
class AbstractSampler(object):
    def __init__(self, unlabeled_data_list_files):
        self.unlabeled_data_list_files = unlabeled_data_list_files

class RandomSampler(AbstractSampler):
    def sample(self, number):
        shuffle(self.unlabeled_data_list_files)
        random_items = []
        for item in self.unlabeled_data_list_files:
            random_items.append(item)
            if len(random_items) >= number:
                break

        return random_items

class LowConfUnlabeledSampler(AbstractSampler):
    def __init__(self, unlabeled_data_list_files):
        super().__init__(unlabeled_data_list_files) # call parent init

    def sample(self, model, number):
        if model is None:
            raise Exception("model cannot be empty")
        
        confidences = []
        for image_path in self.unlabeled_data_list_files:
            img = image.load_img(img_path, target_size=(50, 50))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_batch)
            print("prediction :", prediction)

            if prob_related < 0.5:
                confidence = 1 - prob_related
            else:
               confidence = prob_related
            item = [image_path, confidence]
            confidences.append(item)
        confidences.sort(key=lambda x: x[1])
        return confidences[:number:]
            

class SamplingProcessor(object):
    def __init__(self, run, config):
        self.run = run
        self.config = config
    
    def process(self, input_data,register_model_folder, sampled_data):
        print('salut')

if __name__ == "__main__":

    run = Run.get_context()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data', help='prepped data folder mounting point')
    parser.add_argument('--sampled_data', type=str, dest='sampled_data', help='model candidate destination folder mounting point')
    parser.add_argument('--registered_model_folder', type=str, dest='registered_model_folder', help='model location')
    parser.add_argument('--mode', type=str, dest="mode")
    args = parser.parse_args()
    
    
    input_data = args.input_data
    sampled_data = args.sampled_data
    registered_model_folder = args.registered_model_folder
    mode = args.mode
    if mode == "execute":
        configHandler = ConfigHandler()
        config = configHandler.get_file("config.yaml")

        sampler = SamplingProcessor(run, config)
        sampler.process(input_data,registered_model_folder, sampled_data)
    else:
        print("the mode has value '{0}' so no need to execute data sampling step".format(mode))