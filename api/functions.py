# more info on subprocesses here: https://docs.python.org/3/library/subprocess.html
import subprocess
import json

# 3rd party modules
from flask import abort, make_response

def refresh_vcdb():
	bashCommand = "./run_app/refresh_vcdb.sh"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if "failed" in process.output:
		abort(
            401, f" Failed with error: {process.stderr}"
        )
	return make_response(
			f"Successfully cloned current VCDB version", 200
    		)	

def get_running_models():
	with open('../config.json') as json_file:
		config = json.load(json_file)
		ports_models = config['ports_models']

	open_ports_models = {}
	for port, model_name in ports_models.items():
		bashCommand = f'./run_app/discover_{model_name}'
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		if process.output != None:
			open_ports_models[port] = model_name
		
	return make_response(
			json.dumps(open_ports_models), 200
    		)	

def retrain_model(model_name):
	# here I need as a parameter the name or port of service or 'all' to train em all
	# I also need to kill the selected models first !!!!
	bashCommand = "./run_app/kill.sh"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if process.output != None:
		abort(
            401, f" Failed with error: {process.stderr}"
        )
	return make_response(
			f"Successfully stopped serving models", 200
    		)

# first way to run a bash command: simply with .run()
# does the job almost for all cases. 

# It returns a CompletedProcess object which has various methods 
# which allow you to retrieve the exit status, the standard output, 
# and a few other results and status indicators from the finished subprocess.
def fun_ls():
	bashCommand = "ls"
	process = subprocess.run(bashCommand)

	if process.stderr != None:
		abort(
            401, f"ls failed with error: {process.stderr}"
        )
	return make_response(
			f"ls run like a charm", 200
    		)

# second way to run: with subprocess.Popen. 
# process.communicate captures the results

def fun_pwd():
	bashCommand = "pwd"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if process.stderr != None:
		abort(
            401, f"pwd failed with error: {process.stderr}"
        )
	return make_response(
			f"pwd run like a charm", 200
    		)

def fun(command):
	bashCommand = command
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	if error != None:
		abort(
            401, f"{command} failed with error: {error}"
        )
	return make_response(
			f"{command} run like a charm", 200
    		)

if __name__ == "__main__":
	fun_ls()
	fun_pwd()
