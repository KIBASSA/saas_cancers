import os
import sys 
script_path = os.path.join(os.getcwd(), "../utils")
sys.path.append(script_path)
from global_helpers import FilesProviders, ConfigHandler, WorkspaceProvider, ComputeTargetConfig, DataStoreConfig, EndpointPipelinePublisher, LogicAppPipelineConfigManager
from azureml.core import Workspace,Experiment
from azureml.core.dataset import Dataset
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import PipelineEndpoint
from azureml.train.estimator import Estimator
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.graph import PipelineParameter
import shutil
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
script_folder = 'diagnoz_data_science_scripts_pipeline'

try:
    generated_config_file = "../../config.yaml" 
    configHandler = ConfigHandler()
    config = configHandler.get_file(generated_config_file)

    #provide workspace by config file
    workspaceProvider = WorkspaceProvider(config)
    ws,_ = workspaceProvider.get_ws()

    experiment = Experiment(workspace=ws, name=config.EXPERIMENT_PREP_NAME)
    print("experiment {0} created".format(config.EXPERIMENT_PREP_NAME))

    # ----COMPUTE TARGET------   
    #-------------------------
    compute_target = ComputeTargetConfig.config_create(ws,config.AML_COMPUTE_PREP_CLUSTER_NAME,
                                                    config.AML_COMPUTE_PREP_CLUSTER_VM_TYPE,
                                                    config.AML_COMPUTE_CLUSTER_MIN_NODES,
                                                    config.AML_COMPUTE_CLUSTER_MAX_NODES,
                                                    config.IDLE_SECONDS_BEFORE_SCALEDOWN)
    
    blob_datastore = DataStoreConfig.config(ws,config.BLOB_DATASTORE_NAME,
                                            config.ACCOUNT_NAME,
                                            config.CONTAINER_HUML_NAME,
                                            config.ACCOUNT_KEY)

    print("get datasets from datastore")

    input_data_paths =[(blob_datastore, 'mldata')]
    input_dataset = Dataset.File.from_files(path=input_data_paths)


    # ----PYTHON ENV------   
    #-------------------------
    packages = CondaDependencies.create(conda_packages=["cudatoolkit=10.0"],
                                          pip_packages=['azureml-sdk',
                                                            'PyYAML', 
                                                              'azure-storage-blob',
                                                                 'matplotlib',
                                                                 'seaborn',
                                                                 'tensorflow',
                                                                   'Keras',
                                                                      'tensorflow-hub',
                                                                        'joblib',
                                                                         'tqdm',
                                                                         'Pillow',
                                                                          'azureml-dataprep[pandas,fuse]>=1.1.14'])

    diagnoz_env = Environment("diagnoz-pipeline-env")
    diagnoz_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
    diagnoz_env.docker.enabled = True # Use a docker container
    diagnoz_env.docker.base_image = DEFAULT_GPU_IMAGE
    diagnoz_env.python.conda_dependencies = packages
    diagnoz_env.register(workspace=ws)

    # Runconfigs
    pipeline_run_config = RunConfiguration()
    pipeline_run_config.target = compute_target
    pipeline_run_config.environment = diagnoz_env
    print ("Run configuration created.")

    shutil.rmtree(script_folder, ignore_errors=True)
    os.makedirs(script_folder, exist_ok=True)

    #copy all necessary scripts
    files = FilesProviders.get_path_files("../", [os.path.basename(__file__),"__init__.py"])

    for f in files:
      shutil.copy(f, script_folder)
    #add generated config file to script folder
    shutil.copy(generated_config_file, script_folder)

    input_data = input_dataset.as_named_input('input_dataset').as_mount()

    data_store = ws.get_default_datastore()
    prepped_data = PipelineData('prepped_data',  datastore=data_store)

    pipeline_mode_param = PipelineParameter(name="mode",default_value="deploy")

    prep_step = PythonScriptStep(name = 'Prepare data',
                          source_directory = script_folder,
                          script_name = 'prep_data.py',
                          compute_target = compute_target,
                          runconfig = pipeline_run_config,
                          # Specify dataset as initial input
                          inputs=[input_data],
                          # Specify PipelineData as output
                          outputs=[prepped_data],
                          # Also pass as data reference to script
                          arguments = ['--input_data',input_data,
                                        '--prepped_data', prepped_data,
                                        '--mode', pipeline_mode_param], 
                          allow_reuse=False)

    # Construct the pipeline
    pipeline_steps = [prep_step]
    #pipeline_steps = [step_test]
    pipeline = Pipeline(workspace = ws, steps=pipeline_steps)
    print("Pipeline is built.")

    # Create an experiment and run the pipeline
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted for execution.")

    pipeline_run.wait_for_completion()

    #Publish the pipeline and its endpoint
    publisher = EndpointPipelinePublisher(ws)
    published_endpoint = publisher.publish(config.EXPERIMENT_PREP_NAME,pipeline, config.PIPELINE_PREP_NAME, config.PIPELINE_PREP_ENDPOINT)

    #publish pipeline config for logic app
    logicappManager = LogicAppPipelineConfigManager(config)
    logicappManager.update("-",published_endpoint,"published_prep_pipeline.json")

except Exception as e:
  raise Exception(e)
finally:
  shutil.rmtree(script_folder, ignore_errors=True)