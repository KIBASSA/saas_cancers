import os
import sys 
script_path = os.path.join(os.getcwd(), "../utils")
sys.path.append(script_path) 
from global_helpers import FilesProviders, ConfigHandler, WorkspaceProvider, ComputeTargetConfig, DataStoreConfig, EndpointPipelinePublisher, LogicAppPipelineConfigManager  # pylint: disable=import-error
from azureml.core import Workspace,Experiment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import PipelineEndpoint
from azureml.train.estimator import Estimator
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core.runconfig import RunConfiguration
import shutil
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment
script_folder = 'diagnoz_data_science_scripts_pipeline'

try:
    generated_config_file = "../../config.yaml" 
    configHandler = ConfigHandler()
    config = configHandler.get_file(generated_config_file)

    #provide workspace by config file
    workspaceProvider = WorkspaceProvider(config)
    ws,_ = workspaceProvider.get_ws()

    experiment = Experiment(workspace=ws, name=config.EXPERIMENT_DS_NAME)
    print("experiment {0} created".format(config.EXPERIMENT_DS_NAME))

    # ----COMPUTE TARGET------   
    #-------------------------
    compute_target = ComputeTargetConfig.config_create(ws,config.AML_COMPUTE_DS_CLUSTER_NAME,
                                                    config.AML_COMPUTE_DS_CLUSTER_VM_TYPE,
                                                    config.AML_COMPUTE_CLUSTER_MIN_NODES,
                                                    config.AML_COMPUTE_CLUSTER_MAX_NODES,
                                                    config.IDLE_SECONDS_BEFORE_SCALEDOWN)
    
    print("config.BLOB_DATASTORE_NAME : ", config.BLOB_DATASTORE_NAME)
    print("config.ACCOUNT_NAME :",  config.ACCOUNT_NAME)
    print("config.CONTAINER_NAME :", config.CONTAINER_NAME)
    print("config.ACCOUNT_KEY :", config.ACCOUNT_KEY)
    blob_datastore = DataStoreConfig.config(ws,config.BLOB_DATASTORE_NAME,
                                                    config.ACCOUNT_NAME,
                                                        config.CONTAINER_NAME,
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
                                                                   'keras-metrics',
                                                                      'scikit-learn',
                                                                      'tensorflow-hub',
                                                                        'joblib',
                                                                         'tqdm',
                                                                         "imblearn",
                                                                         'Pillow',
                                                                           'opencv-python',
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

    """This parameter allows to specify that either the Pipeline execution will only
    be in "deploy" mode (just publish the code) or in "execute" mode (execute the pipeline code).
    """

    pipeline_mode_param = PipelineParameter(name="mode",default_value="execute")

    estimator_train = Estimator(source_directory=script_folder,
                          compute_target = compute_target,
                          environment_definition=pipeline_run_config.environment,
                          entry_script='train.py')

    model_candidate_folder = PipelineData('model_candidate_folder',  datastore = data_store)

    # Step to run an estimator
    train_step = EstimatorStep(name = 'Train model',
                            estimator = estimator_train,
                            compute_target = compute_target,
                            # Specify PipelineData as input
                            inputs=[input_data],
                            outputs=[model_candidate_folder],
                            # Pass as data reference to estimator script
                            estimator_entry_script_arguments=['--input_data',input_data,
                                                              '--model_candidate_folder',model_candidate_folder,
                                                              '--mode', pipeline_mode_param], 
                            allow_reuse=False)
    

    estimator_evaluate = Estimator(source_directory=script_folder,
                          compute_target = compute_target,
                          environment_definition=pipeline_run_config.environment,
                          entry_script='eval_model.py')

    validated_model_folder = PipelineData('validated_model_folder',  datastore = data_store)

    # Step 4 to run a Python script
    evaluate_step = EstimatorStep(name = 'Evaluate model',
                            estimator = estimator_evaluate,
                            compute_target = compute_target,
                            # Specify PipelineData as input
                            inputs=[model_candidate_folder,input_data],
                            outputs=[validated_model_folder],
                            # Pass as data reference to estimator script
                            estimator_entry_script_arguments=['--input_data',input_data,
                                                            '--model_candidate_folder',model_candidate_folder,
                                                            '--validated_model_folder',validated_model_folder,
                                                            '--mode', pipeline_mode_param], 
                            allow_reuse=False)

    # Step 3, run the model registration script
    registered_model_folder = PipelineData('registered_model_folder',  datastore = data_store)
    register_step = PythonScriptStep(name = "Register Model",
                                    source_directory = script_folder,
                                    script_name = "register.py",
                                    arguments = ['--validated_model_folder', validated_model_folder,
                                                '--registered_model_folder', registered_model_folder,
                                                '--mode', pipeline_mode_param],
                                    inputs=[validated_model_folder],
                                    outputs=[registered_model_folder],
                                    compute_target = compute_target,
                                    runconfig = pipeline_run_config, 
                            allow_reuse=False)


    estimator_sampling = Estimator(source_directory=script_folder,
                        compute_target = compute_target,
                        environment_definition=pipeline_run_config.environment,
                        entry_script='sampling.py')

    sampled_data = PipelineData('sampled_data',  datastore=data_store)
    # Step 4 to run a Python script
    sampling_step = EstimatorStep(name = 'Sampling',
                        estimator = estimator_sampling,
                        compute_target = compute_target,
                        # Specify PipelineData as input
                        inputs=[input_data, registered_model_folder],
                        outputs=[sampled_data],
                        # Pass as data reference to estimator script
                        estimator_entry_script_arguments=['--input_data', input_data,
                                                            '--registered_model_folder',registered_model_folder,
                                                            '--sampled_data', sampled_data,
                                                            '--mode', pipeline_mode_param], 
                            allow_reuse=False)


    # Construct the pipeline
    pipeline_steps = [train_step, evaluate_step, register_step, sampling_step]
    #pipeline_steps = [step_test]
    pipeline = Pipeline(workspace = ws, steps=pipeline_steps)
    print("Pipeline is built.")

    # Create an experiment and run the pipeline
    pipeline_run = experiment.submit(pipeline)
    print("Pipeline submitted for execution.")

    pipeline_run.wait_for_completion()

    #Publish the pipeline and its endpoint
    publisher = EndpointPipelinePublisher(ws)
    published_endpoint = publisher.publish(config.EXPERIMENT_DS_NAME,pipeline, config.PIPELINE_DS_NAME, config.PIPELINE_DS_ENDPOINT)

    #publish pipeline config for logic app
    logicappManager = LogicAppPipelineConfigManager(config)
    logicappManager.update("-",published_endpoint,"published_ds_pipeline.json")

except Exception as e:
  raise Exception(e)
finally:
  #print("ddd")
  shutil.rmtree(script_folder, ignore_errors=True)