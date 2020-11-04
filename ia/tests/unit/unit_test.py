
import os
import sys
import glob
from distutils.dir_util import copy_tree
import shutil
script_path = os.path.join(os.path.dirname(__file__), "../../src/scripts")
sys.path.append(script_path)
script_path = os.path.join(os.path.dirname(__file__), "../../src/utils")
sys.path.append(script_path)
script_path = os.path.join(os.path.dirname(__file__), "../../src/models/Gans/DCGAN")
sys.path.append(script_path)
script_path = os.path.join(os.path.dirname(__file__), "../../src/models/processors")
sys.path.append(script_path)
from prep_data import DataMerger, DataPreparator
from train import ModelTrainer
from eval_model import ModelValidator
from register import ModelRegister
from sampling import RandomSampler, LowConfUnlabeledSampler, SamplingProcessor
from global_helpers import AzureMLLogsProvider, ConfigHandler
from discriminator import disc_network

"""|||||||||| DataMerger ||||||||||
Testing the useful functions of the DataMerger class.
"""

def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class ImagePathListUploaderMoq(object):
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager
    
    def upload(self,files_source, blob_container_dest):
        print("upload")

class BlobManagerMoq(object):
    def upload(self, blob_container, file_path, overwrite = False):
        print("upload")
    
class DataUploaderMoq(object):
    def __init__(self, blob_manager):
        self.blob_manager = blob_manager

    def upload(self,folder_source, blob_container_dest):
        print("upload")

class AzureExperimentMoq:
    def __init__(self):
        self.runs = []

    def get_runs(self):
        return self.runs
        
class WeorkspaceMoq:
    def __init__(self):
        print("WeorkspaceMoq")

class ExperimentMoq:
    def __init__(self):
        self.workspace = WeorkspaceMoq()

class AzureMLRunMoq:
    def __init__(self, parent):
        self.parent = parent

        self.experiment = ExperimentMoq()

        self.children = []
        self.json_data = {}
        self.tags = {}
        self.metrics = {}

    def get_children(self):
        return self.children
    
    def get_details(self):
        return self.json_data

    def get_tags(self):
        return self.tags
    
    def get_metrics(self):
        return self.metrics
    
    def tag(self, title, value):
        print("title :", title)
        print("value :", value)

    def log(self, title, value):
        print("title :", title)
        print("value :", value)

    def log_image(self, title, plot):
        print("tile:", title)
        print("plot :", plot)
        id = Helper.generate()
        if self.temp_dir:
            plot.savefig(os.path.join(self.temp_dir,"{0}_{1}.png".format(title, id)))
    
    def upload_file(self, name, path_or_stream):
        print("name :", name)
        print("path_or_stream :", path_or_stream)
    
    def set_temp_dir(self, temp_dir):
        self.temp_dir = temp_dir
    
    def register_model(self, model_name,tags,model_path):
        print("model_name : ", model_name)
        print("model_path : ", model_path)

class WebServiceDeployerMoq:
    def __init__(self, ws, config):
        self.ws = ws
        self.config = config

    def to_deploy(self, principal_metric_value):
        return principal_metric_value >= self.config.DEPLOY_THRESHOLD
    
    def deploy(self):
        print("deployed")

def test_if_method___merge__is_running_properly():
    annotated_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/annotated")
    train_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/train")
    train_result_folder = os.path.join(os.path.dirname(__file__), "data/data_merger/merge/train_result")
    
    clean_dir(train_result_folder)
    copy_tree(train_folder, train_result_folder)

    data_merger = DataMerger()

    annotated_folder_label_0 = os.path.join(annotated_folder, "0")
    train_folder_label_0 = os.path.join(train_result_folder, "0")
    data_merger.merge(annotated_folder_label_0, train_folder_label_0)
    
    train_label_0_files = glob.glob(train_folder_label_0 + '/*.png')
    assert len(train_label_0_files) == 9

    annotated_folder_label_1 = os.path.join(annotated_folder, "1")
    train_folder_label_1 = os.path.join(train_result_folder, "1")
    data_merger.merge(annotated_folder_label_1, train_folder_label_1)

    train_folder_label_1 = glob.glob(train_folder_label_1 + '/*.png')
    assert len(train_folder_label_1) == 9

"""|||||||||| DataPreparator ||||||||||
Testing the useful functions of the DataPreparator class.
"""

def test_if_method___prepare__to_eval_folder____is_running_properly():
    #input_data
    input_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_eval_folder")
    prepped_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_eval_folder")
    eval_folder = os.path.join(prepped_data,"diagnoz/mldata/eval_data")
    if os.path.isdir(eval_folder):
        clean_dir(eval_folder)
    shutil.rmtree(eval_folder, ignore_errors=True)
    
    data_merger = DataMerger()
    blob_manager = BlobManagerMoq()
    data_uploader = DataUploaderMoq(blob_manager)
    run = AzureMLRunMoq(None)
    data_peparator = DataPreparator(run)
    data_peparator.prepare(input_data,prepped_data,data_merger,data_uploader)

    eval_folder_label_0 = os.path.join(eval_folder,"0")
    eval_label_0_files = glob.glob(eval_folder_label_0 + '/*.png')
    assert len(eval_label_0_files) == 7
    eval_folder_label_1 = os.path.join(eval_folder,"1")
    eval_label_1_files = glob.glob(eval_folder_label_1 + '/*.png')
    assert len(eval_label_1_files) == 7

def test_if_method___prepare__to_train_folder____is_running_properly():
    #input_data
    input_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_train_folder")
    prepped_data = os.path.join(os.path.dirname(__file__),"data/data_preparator/prepare___to_train_folder")
    train_folder = os.path.join(prepped_data,"diagnoz/mldata/train_data")
    if os.path.isdir(train_folder):
        clean_dir(train_folder)
    shutil.rmtree(train_folder, ignore_errors=True)
    
    data_merger = DataMerger()
    blob_manager = BlobManagerMoq()
    data_uploader = DataUploaderMoq(blob_manager)
    run = AzureMLRunMoq(None)
    data_peparator = DataPreparator(run)
    data_peparator.prepare(input_data,prepped_data,data_merger,data_uploader)

    train_folder_label_0 = os.path.join(train_folder,"0")
    train_label_0_files = glob.glob(train_folder_label_0 + '/*.png')
    assert len(train_label_0_files) == 7
    train_folder_label_1 = os.path.join(train_folder,"1")
    train_label_1_files = glob.glob(train_folder_label_1 + '/*.png')
    assert len(train_label_1_files) == 7


"""|||||||||| ModelTrainer ||||||||||
Testing the useful functions of the ModelTrainer class.
"""

def test_if_ModelTrainer_method___train____is_running_properly():
    input_data = prepped_data = os.path.join(os.path.dirname(__file__),"data/model_trainer/train")
    model_candidate_folder = os.path.join(prepped_data,"diagnoz/mldata/models")
    if os.path.isdir(model_candidate_folder):
        clean_dir(model_candidate_folder)

    run = AzureMLRunMoq(None)
    trainer = ModelTrainer(run)
    trainer.train(input_data, prepped_data, model_candidate_folder)
    classifier_file = os.path.join(model_candidate_folder, "classifier.hdf5")
    assert os.path.isfile(classifier_file) == True
    generator_file = os.path.join(model_candidate_folder, "generator.hdf5")
    assert os.path.isfile(generator_file) == True


"""|||||||||| ModelValidator ||||||||||
Testing the useful functions of the ModelValidator class.
"""

def test_if_ModelTrainer___evaluate___is_running_properly():
    input_data = os.path.join(os.path.dirname(__file__),"data/model_validator/evaluate")
    model_candidate_folder = os.path.join(input_data,"diagnoz/mldata/model_candidate")
    validated_model_folder = os.path.join(input_data,"diagnoz/mldata/validated_model")
    if os.path.isdir(validated_model_folder):
        clean_dir(validated_model_folder)
        shutil.rmtree(validated_model_folder, ignore_errors=True)
    
    parent = AzureMLRunMoq(None)
    child = AzureMLRunMoq(parent)
    child.json_data = {"runDefinition": {"script": "prep_data.py"}}
    child.tags = {"IGNORE_TRAIN_STEP":False}
    parent.children.append(child)
    run = AzureMLRunMoq(parent)
    azure_ml_logs_provider = AzureMLLogsProvider(run)
    model_validator = ModelValidator(run, azure_ml_logs_provider)
    model_validator.evaluate(input_data,model_candidate_folder, validated_model_folder)
    classifier_file = os.path.join(validated_model_folder, "classifier.hdf5")
    assert os.path.isfile(classifier_file) == True


"""|||||||||| ModelRegister ||||||||||
Testing the useful functions of the ModelRegister class.
"""
def test_if_ModelRegister___register___is_running_properly():
    validated_model_folder = os.path.join(os.path.dirname(__file__),"data/model_register/register/validated_model")
    registered_model_folder = os.path.join(os.path.dirname(__file__),"data/model_register/register/registered_model")
    if os.path.isdir(registered_model_folder):
        clean_dir(registered_model_folder)
        shutil.rmtree(registered_model_folder, ignore_errors=True)
    
    parent = AzureMLRunMoq(None)
    child = AzureMLRunMoq(parent)
    child.json_data = {"runDefinition": {"script": "prep_data.py"}}
    child.tags = {"IGNORE_TRAIN_STEP":False}
    parent.children.append(child)

    child = AzureMLRunMoq(parent)
    child.json_data = {"runDefinition": {"script": "eval_model.py"}}
    child.metrics = {"acc":0.8}
    parent.children.append(child)
    
    run = AzureMLRunMoq(parent)
    azure_ml_logs_provider = AzureMLLogsProvider(run)

    configHandler = ConfigHandler()
    config = configHandler.get_file(os.path.join(os.path.dirname(__file__),"data/model_register/register/config.yaml"))
    model_deployer = WebServiceDeployerMoq(run, config)
    model_register = ModelRegister(run, config)
    model_register.register(validated_model_folder, registered_model_folder, azure_ml_logs_provider, model_deployer)
    classifier_file = os.path.join(registered_model_folder, "classifier.hdf5")
    assert os.path.isfile(classifier_file) == True


"""|||||||||| RandomSampler ||||||||||
Testing the useful functions of the RandomSampler class.
"""
def test_if_RandomSampler___sample___is_running_properly():
    unlabeled_path = os.path.join(os.path.dirname(__file__), "data/random_sampler/sample/unlabeled_data/data")
    unlabeled_images_list = glob.glob(unlabeled_path + '\*.png')
    random_sampler = RandomSampler()
    sampled_images = random_sampler.sample(unlabeled_images_list, 20)
    assert len(sampled_images) == 20
    


"""|||||||||| LowConfUnlabeledSampler ||||||||||
Testing the useful functions of the LowConfUnlabeledSampler class.
"""
def test_if_LowConfUnlabeledSampler___sample___is_running_properly():
    unlabeled_path = os.path.join(os.path.dirname(__file__), "data/low_conf_unlabeled_sampler/sample/unlabeled_data/data")
    unlabeled_images_list = glob.glob(unlabeled_path + '\*.png')
    classifier_file = os.path.join(os.path.dirname(__file__), "data/low_conf_unlabeled_sampler/sample/model_register/classifier.hdf5")
    assert os.path.isfile(classifier_file) == True
    _, classifier = disc_network()
    classifier.load_weights(classifier_file)
    low_conf_sampler = LowConfUnlabeledSampler()
    sampled_images = low_conf_sampler.sample(classifier, unlabeled_images_list, 30)
    assert len(sampled_images) == 30


"""|||||||||| SamplingProcessor ||||||||||
Testing the useful functions of the SamplingProcessor class.
"""
def test_if_SamplingProcessor___sample___is_running_properly():
    input_data = os.path.join(os.path.dirname(__file__),"data/sampling_processor/process")
    registered_model_folder = os.path.join(os.path.dirname(__file__),"data/sampling_processor/process/registered_model")
    sampled_data_folder = os.path.join(os.path.dirname(__file__),"data/sampling_processor/process/sampled_data")
    if os.path.isdir(sampled_data_folder):
        clean_dir(sampled_data_folder)
        shutil.rmtree(sampled_data_folder, ignore_errors=True)
    
    random_sampler = RandomSampler()
    low_conf_sampler = LowConfUnlabeledSampler()
    blob_manager = BlobManagerMoq()
    imagepath_list_uploader = ImagePathListUploaderMoq(blob_manager)
    run = AzureMLRunMoq(None)
    sampler_processor = SamplingProcessor(run)
    sampler_processor.process(input_data, 
                                registered_model_folder,
                                        sampled_data_folder, 
                                                random_sampler, 
                                                        low_conf_sampler, 
                                                                imagepath_list_uploader)

    sampled_data_images = glob.glob(sampled_data_folder + '/*.png')
    assert len(sampled_data_images) == 200

if __name__ == "__main__":
    test_if_method___merge__is_running_properly()
    test_if_method___prepare__to_eval_folder____is_running_properly()
    test_if_method___prepare__to_train_folder____is_running_properly()
    test_if_ModelTrainer_method___train____is_running_properly()
    test_if_ModelTrainer___evaluate___is_running_properly()
    test_if_ModelRegister___register___is_running_properly()
    test_if_RandomSampler___sample___is_running_properly()
    test_if_LowConfUnlabeledSampler___sample___is_running_properly()
    test_if_SamplingProcessor___sample___is_running_properly()