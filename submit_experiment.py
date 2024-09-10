from azureml.core import Workspace, Experiment, ScriptRunConfig

# Load the Azure ML workspace
ws = Workspace.from_config()

# Define the experiment name
experiment_name = "diabetic-detection-experiment"  # Update this name

# Create an experiment
experiment = Experiment(workspace=ws, name=experiment_name)

# Define the configuration for running the training script
script_config = ScriptRunConfig(source_directory='.', script='train.py')

# Submit the experiment
run = experiment.submit(script_config)

# Wait for the run to complete
run.wait_for_completion(show_output=True)
