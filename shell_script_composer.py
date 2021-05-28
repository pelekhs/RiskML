import json
import subprocess
import os
  
# Create run script
with open('config.json') as json_file:
    config = json.load(json_file)
algorithms = config['algorithms']
ports_models = config['ports_models']

# compose run scripts
with open (os.path.join('app', 'run', 'run_all.sh'), 'w') as rsh:
    
    rsh.write('''#!/bin/bash\nsource app/utils.sh\necho "Training models..."\n''')
    
    for port, model_name in ports_models.items():
    
        rsh.write(f'''train_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port}\n''')

        # create kill script for each separate model
        with open (os.path.join('app', 'run', f'run_{model_name}.sh'), 'w') as f:
            
            f.write(f'''#!/bin/bash\nsource app/utils.sh\ntrain_n_serve "{'.'.join(model_name.split('.')[:-1])}" "{model_name.split('.')[-1]}" "{algorithms}" {port}\n''')
            f.close()
    bashCommand = f'chmod +x app/run/'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

rsh.close()

bashCommand = 'chmod +x app/run/run_all.sh'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

# compose kill scripts
with open (os.path.join('app', 'kill', 'kill_all.sh'), 'w') as rsh:
    
    rsh.write('#!/bin/bash\n')

    for port, model_name in ports_models.items():
        # create kill script for all models
        rsh.write(f'''kill -9 $(lsof -t -i:{port})\n''')
        
        # create kill script for each separate model
        with open (os.path.join('app', 'kill', f'kill_{model_name}.sh'), 'w') as f:
            
            f.write(f'''#!/bin/bash\nkill -9 $(lsof -t -i:{port})\n''')
            f.close()
    bashCommand = f'chmod +x app/kill'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

rsh.close()

bashCommand = 'chmod +x app/kill/kill_all.sh'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

# compose discover scripts
for port, model_name in ports_models.items():
    # create list script for each separate model
    with open (os.path.join('app', 'discover', f'discover_{model_name}.sh'), 'w') as f:
        
        f.write(f'''#!/bin/bash\nlsof -t -i:{port}\n''')
        f.close()
bashCommand = f'chmod +x app/discover'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)