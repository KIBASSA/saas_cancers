from azureml.core import Run
import argparse
from simple_trainer import ModelTrainer # pylint: disable=import-error

if __name__ == "__main__":

    run = Run.get_context()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, dest='input_data', help='data folder mounting point')
    parser.add_argument('--model_candidate_folder', type=str, dest='model_candidate_folder', help='model candidate destination folder mounting point')
    parser.add_argument('--mode', type=str, dest="mode")
    args = parser.parse_args()

    input_data = args.input_data
    model_candidate_folder = args.model_candidate_folder
    mode = args.mode
    if mode == "execute":
        """We create the instance of the ModelTrainer class by passing the Run to it 
            and then we launch the training.
        """
        trainer = ModelTrainer()
        trainer.set_params(run, 10)
        trainer.train(input_data, model_candidate_folder)
        
    else:
        print("the mode has value '{0}' so no need to execute training step".format(mode))