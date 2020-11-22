
from azureml.core import Run
from tensorflow.keras.preprocessing import image
from discriminator import disc_network
from random import shuffle
import numpy as np
import math
import os
import glob
import shutil
import ntpath
from global_helpers import  ImagePathListUploader, ConfigHandler, SampedDataDataManager, AnnotatedDataManager, BlobStorageHandler
from os.path import isfile, join
import argparse
from random import shuffle
import os
from enum import Enum

class SamplerType(Enum):
    random = 1
    lowconf = 2

class SamplerFactory(object):
    def get_sampler(self, sampler_type : SamplerType):
        if sampler_type.value == SamplerType.random.value:
            return RandomSampler()
        elif sampler_type.value == SamplerType.lowconf.value:
            return LowConfUnlabeledSampler()

class RandomSampler(object):
    def sample(self, unlabeled_data_list_files, number):
        shuffle(unlabeled_data_list_files)
        random_items = []
        for item in unlabeled_data_list_files:
            random_items.append([item, 0,'random'])
            if len(random_items) >= number:
                break

        return random_items

class LowConfUnlabeledSampler(object):

    def sample(self, model, unlabeled_data_list_files, number):
        if model is None:
            raise Exception("model cannot be empty")
        
        shuffle(unlabeled_data_list_files)
        unlabeled_data_list_files = unlabeled_data_list_files[:20000]
        confidences = []
        for index, image_path in enumerate(unlabeled_data_list_files):
            img = image.load_img(image_path, target_size=(50, 50))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_batch)
            prob_related = prediction[0][1]

            """ if model predicts that it is not cancer then the condifence
            will be the probability of predicting that it is not cancer (1- confidence) 
            Because we used the probability to predict that it is cancer (prediction[0][1])
            """ 
            if prob_related < 0.5:
                confidence = 1.0 - prob_related
            else:
               confidence = prob_related
            item = [image_path, confidence, 'lowconf']
            confidences.append(item)
            #print("element ", index, "/", len(unlabeled_data_list_files), " prob_related :", prob_related, " confidence :", confidence)
        confidences.sort(key=lambda x: x[1])
        #print("confidences :", confidences)
        print("confidences :", confidences[:number:])
        return confidences[:number:]


class SamplingProcessor(object):
    def __init__(self, run):
        self.run = run

    def process(self, 
                    input_data,
                        register_model_folder, 
                            sampled_data, 
                                random_sampler, 
                                    lowfonc_sampler, 
                                        sampled_data_manager,
                                            annotated_data_manager):
        unlabeled_path = os.path.join(input_data, "unlabeled/data")
        unlabeled_images_list = glob.glob(unlabeled_path + '/*.png')
        sampled_images = random_sampler.sample(unlabeled_images_list, 200)
        classifier_name = "classifier.hdf5"
        classifier_file = os.path.join(register_model_folder, classifier_name)
        if os.path.isfile(classifier_file):
            _, classifier = disc_network()
            classifier.load_weights(classifier_file)

            lowconf_sampled_images = lowfonc_sampler.sample(classifier, unlabeled_images_list, 180)
            sampled_images = sampled_images[:20] + lowconf_sampled_images
        
        #for image_path in sampled_images:
        #    image_path_dest = os.path.join(sampled_data, os.path.basename(image_path))
        #    os.makedirs(sampled_data, exist_ok = True)
        #    shutil.copy(image_path, image_path_dest)
        
        shuffle(sampled_images)
        sampled_data_manager.upload_data(sampled_images)

        annotated_file = os.path.join(input_data, "annotated_data/current/annotated_data.json")
        if not os.path.isfile(annotated_file):
            print("No annotation data provided. skip archive step")
            return
        """
        archiving of the current annotation file because there is a new sample file
        """
        print("archive")
        working_dir = os.path.join(sampled_data, "working_dir")
        os.makedirs(working_dir, exist_ok=True)
        annotated_data_manager.archive(working_dir)

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

        random_sampler = RandomSampler()
        low_conf_sampler = LowConfUnlabeledSampler()
        blob_manager = BlobStorageHandler()
        #imagepath_list_uploader = ImagePathListUploader(blob_manager)
        sampled_data_manager = SampedDataDataManager(blob_manager)
        annotated_data_manager = AnnotatedDataManager(blob_manager)
        sampler = SamplingProcessor(run)
        sampler.process(input_data,
                            registered_model_folder, 
                                sampled_data, 
                                    random_sampler, 
                                        low_conf_sampler, 
                                            sampled_data_manager,
                                                annotated_data_manager)
    else:
        print("the mode has value '{0}' so no need to execute data sampling step".format(mode))