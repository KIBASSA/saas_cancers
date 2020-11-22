import sys
sys.path.append("../../scripts")
from datetime import datetime
from train import ModelTrainer, ModelTrainerType, ModelTrainerFactory
from sampling import RandomSampler, LowConfUnlabeledSampler, SamplerType, SamplerFactory
from eval_model import ModelValidateProcessor
from azure_moqs import AzureMLRunMoq
import shutil
from shutil import copyfile
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from enum import Enum
class MergerData(object):
    def merge(self, annoted_data, training_data):
        for item in annoted_data:
            if item in training_data:
                continue
            training_data.append(item)
        return training_data


class TrainFilesCopier(object):
    def copy(self, data, folder):
        for item in data:
            if "_class0" in item:
                dest = os.path.join(folder, "train/0/{0}".format(os.path.basename(item)))
            else:
                dest = os.path.join(folder, "train/1/{0}".format(os.path.basename(item)))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            copyfile(item, dest)

class ModelEngine(object):
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
    
    def set_process_dependencies(self, files_copier, model_trainer, model_evaluator):
        self.files_copier = files_copier
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator

    def process(self, request):

        if request is None:
            raise Exception("request cannot be None")
        
        if len(request.training_data) == 0:
            return 0, [request, None]

        if self.files_copier is None:
            raise Exception("must provide files_copier")
        
        if self.model_trainer is None:
            raise Exception("must provide model_trainer")

        if self.model_evaluator is None:
            raise Exception("must provide model_evaluator")

        self.files_copier.copy(request.training_data, self.data_path)
        
        model = self.model_trainer.train(self.data_path, self.model_path)
        
        metrics = self.model_evaluator.evaluate(self.data_path, self.model_path)

        return metrics, [request, model]

class SamplingEngine(object):
    def __init__(self, sampler_types):
        self.sampler_types = sampler_types

    def set_sampler_factory(self, sampler_factory):
        self.sampler_factory = sampler_factory

    def process(self, ml_result):
        request, model = ml_result

        if self.sampler_factory is None:
            raise Exception("must provide sampler_factory as SamplerFactory class")

        sampled_images = []
        if SamplerType.random in self.sampler_types:
            random_sampler = self.sampler_factory.get_sampler(SamplerType.random)
            sampled_images = random_sampler.sample(request.unlabeled_data, 200)
        
        if model == None:
            return sampled_images

        if SamplerType.lowconf in self.sampler_types:
            lowfonc_sampler = self.sampler_factory.get_sampler(SamplerType.lowconf)
            lowconf_sampled_images = lowfonc_sampler.sample(model, request.unlabeled_data, 180)
            sampled_images = sampled_images[:20] + lowconf_sampled_images

        return sampled_images

    """
    def process(self, ml_result, random_sampler,lowfonc_sampler):
        request, model = ml_result

        sampled_images = random_sampler.sample(request.unlabeled_data, 200)
        if model == None:
            return sampled_images

        lowconf_sampled_images = lowfonc_sampler.sample(model, request.unlabeled_data, 180)
        sampled_images = sampled_images[:20] + lowconf_sampled_images

        return sampled_images
    """

class AnnotationEngine(object):
    def annotate(self, sampled_data, all_annotated_data):
        annotated_data = []
        for annotated_item in all_annotated_data:
            annotated_item_name = os.path.basename(annotated_item)
            for sampled_item in sampled_data:
                if annotated_item_name == os.path.basename(sampled_item[0]):
                    if annotated_item_name not in annotated_data:
                        annotated_data.append(annotated_item)
        all_annotated_data = list(set(all_annotated_data) - set(annotated_data))
        return annotated_data, all_annotated_data

class RequestData(object):
    def __init__(self, all_annotated_data, unlabeled_data, eval_data, training_data):
        self.all_annotated_data = all_annotated_data
        self.unlabeled_data = unlabeled_data
        self.eval_data = eval_data
        self.training_data = training_data

class RequestDataCreator(object):
    def create(self, unlabeled_path, eval_path, unlabeled_data_limit = -1):
        unlabeled_data = glob.glob(unlabeled_path + '/*.png')
        if unlabeled_data_limit != -1:
            unlabeled_data = unlabeled_data[:unlabeled_data_limit]

        print("len(unlabeled_data) :", len(unlabeled_data))
        all_annotated_data = unlabeled_data
        print("len(all_annotated_data) :", len(all_annotated_data))
        eval_data = glob.glob(unlabeled_path + '/*.png', recursive=True)
        print("len(eval_data) :", len(eval_data))
        return RequestData(all_annotated_data,unlabeled_data,eval_data,[])

class PlotData(object):
    def __init__(self, title, data, col):
        self._title = title
        self._data = data
        self._col = col

    @property
    def title(self):
        return self._title
    
    @property
    def data(self):
        return self._data
    
    @property
    def col(self):
        return self._col

class MetricsCollector(object):
    def __init__(self):
        self.accuracies = []
    
    def collect(self, metric):
        self.accuracies.append(metric)
    
    def get_history_metric(self):
        return [PlotData("accuracies",self.accuracies, "accuracies")]


class DataPloter(object):
    def _plot_data(self,engine_type, title, input_data, col):
        """plot lineplot for vector data

        Arguments:
            title {str} -- title that will be displayed on the workspace allows to display the image
            input_data {list} -- list of data to plot in the workspace
            col {str} -- y-axis column
        """
        #local_path = os.path.join(os.getcwd(), self.engine_type)
        local_path = os.path.join(os.getcwd(), engine_type)
        os.makedirs(local_path, exist_ok=True)
        plt.clf() # Clear the current figure
        input_data = np.array(input_data)
        df = pd.DataFrame(data=input_data, columns=[col])
        _ = sns.lineplot(x=df.index, y=col, data=df)
        plt.savefig(os.path.join(local_path, title + '.png'))

    def plot(self, engine_type,  data_to_plot):
        for item in data_to_plot:
            self._plot_data(engine_type, item.title, item.data, item.col)

class ArtefactCleaner(object):
    def __init__(self, principal_folder, artefact_folders_name):
        self.principal_folder = principal_folder
        self.artefact_folders_name = artefact_folders_name
    
    def clean(self):
        for folder_name in self.artefact_folders_name:
            shutil.rmtree(os.path.join(self.principal_folder, folder_name), ignore_errors=True)

class BenchEngine(object):
    def __init__(self, engine_type):
        self.engine_type = engine_type
    def start(self, 
                request, 
                    merger_data, 
                        model_engine,
                            sampling_engine,
                               annotation_engine,
                                    metrics_collector,
                                       data_ploter,
                                           artefact_cleaner):

        annotated_data = []
        while len(request.all_annotated_data) > len(request.training_data):
            # measure process time
            t_start = datetime.now()

            #-- Merge Data
            print("merge annotated to training data")
            request.training_data = merger_data.merge(annotated_data, request.training_data)

            #-- Prep - Training - Eval
            print("launch ml process")
            metrics, ml_result =  model_engine.process(request)

            #-- Sampling Data
            print("sample new data")
            sampled_data = sampling_engine.process(ml_result)
            print("len(sampled_data) :", len(sampled_data))

            #-- Annotate Data
            print("annotate sampled data")
            annotated_data, request.all_annotated_data = annotation_engine.annotate(sampled_data, request.all_annotated_data)
            #print("len(sampled_data) :", len(sampled_data))

            time_by_iteration = (datetime.now() - t_start).total_seconds()

            print("collecte metrics")
            metrics_collector.collect(metrics)

            print("BenchmarkType(self.engine_type.value).name :", BenchmarkType(self.engine_type.value).name)
            data_ploter.plot(BenchmarkType(self.engine_type.value).name, metrics_collector.get_history_metric())
            
        return metrics_collector.get_history_metric()
        
class BenchmarkType(Enum):
    sample_classifier_random_sampling = 1
    sample_classifier_random_lowconf_sampling = 2
    gan_classifier_random_sampling = 3
    gan_classifier_random_lowconf_sampling = 4

class BenchmarkEnginesProvider(object):
    def __init__(self, types):
        self.types = types
    
    def get_engines(self,model_trainer_factory):
        result = []
        if BenchmarkType.sample_classifier_random_sampling in self.types:
            model_trainer = model_trainer_factory.get_trainer(ModelTrainerType.simple_classifier)
            result.append({"type":BenchmarkType.sample_classifier_random_sampling, "classifier":model_trainer, "sampler_types": [SamplerType.random]})
        if BenchmarkType.sample_classifier_random_lowconf_sampling in self.types:
            model_trainer = model_trainer_factory.get_trainer(ModelTrainerType.simple_classifier)
            result.append({"type":BenchmarkType.sample_classifier_random_lowconf_sampling, "classifier":model_trainer, "sampler_types": [SamplerType.random, SamplerType.lowconf]})
        if BenchmarkType.gan_classifier_random_sampling in self.types:
            model_trainer = model_trainer_factory.get_trainer(ModelTrainerType.gan_classifier)
            result.append({"type":BenchmarkType.gan_classifier_random_sampling, "classifier":model_trainer, "sampler_types": [SamplerType.random]})
        if BenchmarkType.gan_classifier_random_lowconf_sampling in self.types:
            model_trainer = model_trainer_factory.get_trainer(ModelTrainerType.gan_classifier)
            result.append({"type":BenchmarkType.gan_classifier_random_lowconf_sampling, "classifier":model_trainer, "sampler_types": [SamplerType.random, SamplerType.lowconf]})
        return result



if __name__ == "__main__":
    run = AzureMLRunMoq(None)
    benchmark_types = [e for e in BenchmarkType]
    benchmark_engine_provider = BenchmarkEnginesProvider(benchmark_types)
    model_trainer_factory = ModelTrainerFactory()
    items = benchmark_engine_provider.get_engines(model_trainer_factory)
    history_metrics = {}
    for item in items:
        artefact_cleaner =  ArtefactCleaner("../data/", ["train", "models"])
        print("#######type :", item["type"])
        type_name = BenchmarkType(item["type"].value).name
        try:
            request = RequestDataCreator().create("../data/unlabeled/data", "../data/eval")
            merger_data = MergerData()
            model_engine = ModelEngine("../data", "../models")
            trainer = item["classifier"]
            trainer.set_params(run, epochs=1000)
            model_engine.set_process_dependencies(TrainFilesCopier(),trainer,ModelValidateProcessor())
            sampling_engine = SamplingEngine(item["sampler_types"])
            sampling_engine.set_sampler_factory(SamplerFactory())
            annotation_engine = AnnotationEngine()
            metrics_collector = MetricsCollector()
            data_ploter = DataPloter()
            engine = BenchEngine(item["type"])
            plot_data = engine.start(request, 
                                            merger_data, 
                                                model_engine,
                                                    sampling_engine,
                                                        annotation_engine,
                                                            metrics_collector,
                                                                data_ploter, 
                                                                    artefact_cleaner)
            print("history_metric : ", plot_data[0].data)
            history_metrics[type_name] = plot_data[0].data
        except Exception as e:
            raise Exception(e)
        finally:
            """ Clean all afterfact folders
            """
            artefact_cleaner.clean()

    for item in  history_metrics:
        plt.plot(history_metrics[item], label=item)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), "global_report.png"))
        
    #plot global history_metric    
    label = str(input("n'importe quelle touche pour quitter"))
    print("bye")