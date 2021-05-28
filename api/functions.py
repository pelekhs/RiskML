# more info on subprocesses here: https://docs.python.org/3/library/subprocess.html
import subprocess

# 3rd party modules
from flask import abort, make_response


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
