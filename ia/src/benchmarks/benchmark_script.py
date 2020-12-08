from azure_moqs import AzureMLRunMoq
from benchmark_engine import BenchmarkEngineProcessor
if __name__ == "__main__":
    benchmark_processor = BenchmarkEngineProcessor(AzureMLRunMoq(None))
    #process(self, data_path, unlabeled_data_path, eval_data_path, model_path, artefact_cleaner):
    artefact_cleaner =  ArtefactCleaner("../data/", ["train", "models"])
    benchmark_processor.process("../data/",
                                  "../data/unlabeled/data",
                                    "../data/eval",
                                      "../models",
                                        artefact_cleaner)
    