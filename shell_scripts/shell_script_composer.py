import json
import subprocess
import os
  

script_folder = os.path.dirname(os.path.realpath(__file__))

# Create run script
with open(f'{script_folder}/../config.json') as json_file:
    config = json.load(json_file)
algorithms = config['algorithms']
ports_models = config['ports_models']

# Name of folder to store scripts

# Compose init training script to run during image creation
with open (os.path.join(script_folder, 'init_models.sh'), 'w') as rsh:
    
    rsh.write(f'''#!/bin/bash\nsource {script_folder}/train_n_serve.sh\necho "Training initial models..."\n''')
    
    for port, model_name in ports_models.items():
    
        rsh.write(f'''train_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port} "--training" "--serving"\n''')

    rsh.write('''tail -F "KeepingContainerAlive"''')

bashCommand = f'chmod -R 755 {script_folder}/init_models.sh'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

# Compose init training script to run during image run
# with open (os.path.join(script_folder, 'serving.sh'), 'w') as rsh:
    
#     rsh.write(f'''#!/bin/bash\nsource ./train_n_serve.sh\necho "Serving trained models..."\n''')
    
#     for port, model_name in ports_models.items():
    
#         rsh.write(f'''train_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port} "--no-training" "--serving"\n''')

#     rsh.write('''tail -F "KeepingContainerAlive"''')

# bashCommand = f'chmod -R 755 {script_folder}/serving.sh'
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)


# Compose run scripts
with open (os.path.join(script_folder, 'run', 'run_all.sh'), 'w') as rsh:
    
    rsh.write(f'''#!/bin/bash\ncurdir=$( cd "$(dirname "${{BASH_SOURCE[0]}}")" ; pwd -P )\nsource "${{curdir}}"/../train_n_serve.sh\necho "Training models..."\n''')
    
    for port, model_name in ports_models.items():
    
        rsh.write(f'''train_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port}\n''')

        # create run script for each separate model
        with open (os.path.join(script_folder, 'run', f'run_{model_name}.sh'), 'w') as f:
            
            f.write(f'''#!/bin/bash\ncurdir=$( cd "$(dirname "${{BASH_SOURCE[0]}}")" ; pwd -P )\nsource "${{curdir}}"/../train_n_serve.sh\ntrain_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port}\n''')
            f.close()  
    rsh.write('''tail -F "KeepingContainerAlive"''')
    bashCommand = f'chmod -R 755 {script_folder}/run/'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

bashCommand = f'chmod -R 755 {script_folder}/run/run_all.sh'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

# Compose kill scripts
with open (os.path.join(script_folder, 'kill', 'kill_all.sh'), 'w') as rsh:
    
    rsh.write('#!/bin/bash\n')

    for port, model_name in ports_models.items():
        # create kill script for all models
        rsh.write(f'''kill -9 $(lsof -t -i:{port})\n''')
        
        # create kill script for each separate model
        with open (os.path.join(script_folder, 'kill', f'kill_{model_name}.sh'), 'w') as f:
            
            f.write(f'''#!/bin/bash\nkill -9 $(lsof -t -i:{port})\n''')
            f.close()
    bashCommand = f'chmod -R 755 {script_folder}/kill'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

rsh.close()

bashCommand = f'chmod -R 755 {script_folder}/kill/kill_all.sh'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

# Compose discover scripts
for port, model_name in ports_models.items():
    # create list script for each separate model
    with open (os.path.join(script_folder, 'discover', f'discover_{model_name}.sh'), 'w') as f:
        
        f.write(f'''#!/bin/bash\nlsof -t -i:{port}\n''')
        f.close()
bashCommand = f'chmod -R 755 {script_folder}/discover'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

