# more info on subprocesses here: https://docs.python.org/3/library/subprocess.html
import subprocess
import json

# 3rd party modules
from flask import abort, make_response

def updateVCDB():
	""" 
    API function to update the VCDB dataset by cloning the VCDB github
	repo from scratch

    Parameters
    ---------- 
	-

    Returns
    ---------- 
    HTTP response
    
	"""
	bashCommand = "./shell_scripts/updateVCDB.sh"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if "failed" in str(output) or "Error" in str(output):
		abort(
            401, 'Error updating VCDB'
	    	)

	if error != None:
		abort(
            401, f'Failed to kill serving process with error: {error1}'
        	)
	
	return make_response(
			output, 200
    		)	

def getModels():
	""" 
    API function to get the state of currently served models

    Parameters
    ---------- 
	-

    Returns
    ---------- 
    HTTP response
    
	"""
	with open('config.json') as json_file:
		config = json.load(json_file)
		ports_models = config['ports_models']
	
	open_ports_models = {}
	
	for port, model_id in ports_models.items():
		
		bashCommand = f'./shell_scripts/discover/discover_{model_id}.sh'
		try:
			process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
			output, error = process.communicate()
		except FileNotFoundError:
			abort(
            404, 'Model not found'
        	)
		
		if output != b'':
			open_ports_models[str(port)] = model_id
		elif error != None:
			abort(
            401, f"ls failed with error: {process.stderr}"
        	)
		else:
			open_ports_models[str(port)] = 'No service'
	
	return make_response(
			json.dumps(open_ports_models), 200
    		)	

def retrainModels(model_id):
	""" 
    API function to trigger model retraining by model_id or all.

    Parameters
    ---------- 
	model_id: str
		The name of the target VCDB column

    Returns
    ---------- 
    HTTP response
    
	"""

	bashCommand = f"./shell_scripts/kill/kill_{model_id}.sh"
	try:
		process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
		output1, error1 = process.communicate()
	except FileNotFoundError:
			abort(
            	404, 'Kill script not found. Probably model name is not correct!'
        		)
	
	bashCommand = f"./shell_scripts/run/run_{model_id}.sh"
	try:
		process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
		output1, error1 = process.communicate()
	except FileNotFoundError:
		abort(
            404, 'Run script not found'
        	)
	process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
	output2, error2 = process.communicate()

	if error1 != None:
		abort(
            401, f'Failed to kill serving process with error: {error1}'
        	)
	
	if error2 != None:
		abort(
			402, f'Failed to train model with error: {error2}'
		)
	
	return make_response(
			f"Successfully trained models", 200
    		)