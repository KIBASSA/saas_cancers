from azureml.core import Run
import argparse
from simple_trainer import ModelTrainer # pylint: disable=import-error
from benchmark_engine import BenchmarkEngineProcessor # pylint: disable=import-error
import os
if __name__ == "__main__":

    run = Run.get_context()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
    parser.add_argument('--model_folder', type=str, dest='model_folder', help='model candidate destination folder mounting point')
    parser.add_argument('--mode', type=str, dest="mode")
    args = parser.parse_args()

    input_data = args.input_data
    model_folder = args.model_folder
    mode = args.mode
    benchmark_processor = BenchmarkEngineProcessor(run)
    artefact_cleaner =  ArtefactCleaner("../data/", [])
    benchmark_processor.process(input_data,
                                os.path.join(input_data,"unlabeled/data"),
                                 os.path.join(input_data,"eval"),
                                    os.path.join(model_folder, "models"),
                                        artefact_cleaner)