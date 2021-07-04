RiskML: Machine learning models for cyber risk assessment - The VCDB case
=========================================================================

The repository contains python code for training different ML models on the well-known Veris Community Database (VCDB). It is mainly based on the MLflow framework. 

Training
--------
The core of the code is the train.py file which makes use of most other files and functions developed. This script performs all training, evaluation and model registration.  Results are stored in a locally running MLflow server.

* **Prerequisites**: <br> 
python3, pip & ``pip install -r requirements.txt``

* **Run**: <br>
``mlflow run --experiment-name <experiment_name_for_mlflow_tracking> --entry-point train . -P task=<task_option> -P target=<target_option> -P algo=<algo_option> -P hyperparams=<hyperparams_option> -P train_size=<train_size_option> -P imputer=<imputer_option> -P pca=<pca_option> -P split_random_state=<split_random_state_option> -P n_folds=<n_folds_option> -P explain=<explain_option> -P merge=<merge_option> -P shap_data_percentage=<shap_data_percentage_option> -P test_over_train_percentage=<test_over_train_percentage_option>  --no-conda``

 * **Run options**: <br>

    -t, --task [attribute|asset.variety|asset.assets.variety.S|asset.assets.variety.M|asset.assets.variety.U|asset.assets.variety.P|asset.assets.variety.T|action|action.error.variety|action.hacking.variety|action.misuse.variety|action.physical.variety|action.malware.variety|action.social.variety|default]
                                  Learning task
  -tt, --target TEXT              Specific target
                                  variable. Omit this
                                  option to get list of
                                  options according to
                                  task

  -a, --algo [SVM|RF|LR|GNB|LGBM|KNN|default]
                                  Algorithm
  -h, --hyperparams TEXT          "Hyperapameters of
                                  algorithm. e.g.
                                  '{"C": 1, "gamma":
                                  0.1}'

  -i, --imputer [dropnan|default]
                                  Imputation strategy
  -ts, --train-size FLOAT         Training set size. If
                                  equals 1 then cross
                                  validation is
                                  performed to evaluate
                                  the models

  -rs, --split-random-state INTEGER
                                  Random state for
                                  splitting train /
                                  test or cv

  -f, --n-folds INTEGER           Number of folds for
                                  CV if there training
                                  set is all dataset

  -p, --pca INTEGER               Number of PCA
                                  components. 0 means
                                  no PCA

  -e, --explain TEXT              Whether to use SHAP
                                  for explanations.
                                  Requires train_size <
                                  1 and it is generally
                                  a slow process.
                                  Accepted values:
                                  ['y', 'yes', 't',
                                  'true', 'on']
                                  and ['n', 'no', 'f',
                                  false', 'off']

  -m, --merge TEXT                Whether to merge
                                  Brute force, SQL
                                  injection and DoS
                                  columns for hacking
                                  and malware cases.
                                  Accepted values:
                                  ['y', 'yes', 't',
                                  'true', 'on']
                                  and ['n', 'no', 'f',
                                  false', 'off']

  -ts, --train-size FLOAT         Training set size. If
                                  equals 1 then cross
                                  validation is
                                  performed to evaluate
                                  the models

  -sdp, --shap-data-percentage FLOAT
                                  Dataset fraction to
                                  be used with SHAP for
                                  explanations

  -sdp, --shap-test-over-train-percentage FLOAT
                                  Training set fraction
                                  to be used as test
                                  with SHAP for
                                  explanations

  --help                          Show this message and
                                  exit.
                    
The entrypoint options are defined in the "MLproject" file along with the click decorators of the train_evaluate_register() function of train.py. 

Unset parameters result to their default values as defined in the "train" entrypoint of the ``MLproject`` file.

The MLflow UI (http://localhost:5000) can be used for tracking model performance metrics. Models, artifacts and tracking parameters are stored locally.

* **Valid run example**: <br>
``mlflow run --experiment-name 'asset.variety' --entry-point train . -P task="asset.variety" -P target="Server" -P algo="LGBM" -P train_size=1 -P split_random_state=44 -P n_folds=5 -P pca=2 -P explain="yes" -P merge="yes"  --no-conda ``

Hyperparameter tuning
---------------------
The project also provides capabilities of hyperparameter tuning on a selected tasks and their recursive targets. The algorithms can be defined by the user in ``models.py`` following the dictionary structure that can already be found there. Results are stored in a locally running MLflow server. 

* **Prerequisites**: <br> 
python3, pip &  ``pip install -r requirements.txt``

* **Run**: <br>
``mlflow run --experiment-name <experiment_name_for_mlflow_tracking> --entry-point gridsearch . -P task=<task_option> -P metric=<metric_option> -P averaging=<averaging_option> -P imputer=<imputer_option> -P train_size=<train_size_option> -P pca=<pca_option> -P random_state=<random_state_option> -P n_folds=<n_folds_option> -P n_jobs_cv=<n_jobs_cv_option> -P explain=<explain_option> -P merge=<merge_option>  --no-conda``

 * **Run options**: <br>

  -t, --task [asset.variety|asset.assets.variety.S|asset.assets.variety.M|asset.assets.variety.U|asset.assets.variety.P|asset.assets.variety.T|action|action.error.variety|action.hacking.variety|action.misuse.variety|action.physical.variety|action.malware.variety|action.social.variety|default]
                                  Learning task
  -i, --imputer [dropnan|default]
                                  Imputation strategy
  -m, --metric [auc|accuracy|precision|recall|f1|hl|default]
                                  Metric to maximise
                                  during tuning.

  -a, --averaging [micro|macro|weighted|default]
                                  Method to compute
                                  aggregate metrics

  -f, --n-folds INTEGER           Number of folds for
                                  CV

  -j, --n-jobs-cv INTEGER         Number of cores for
                                  GridsearchCV

  -r, --random-state INTEGER      Random state for
                                  train/test splits

  -p, --pca INTEGER               Number of PCA
                                  components. 0 means
                                  no PCA

  -m, --merge TEXT                Whether to merge
                                  Brute force, SQL
                                  injection and DoS
                                  columns for hacking
                                  and malware cases.
                                  Accepted values:
                                  ['y', 'yes', 't',
                                  'true', 'on']
                                  and ['n', 'no', 'f',
                                  false', 'off']

  --help                          Show this message and
                                  exit.
                    
The entrypoint options are defined in the "MLproject" file along with the click decorators of the run() function of hyperparameter_tuning/gridsearch.py. 

Unset parameters result to their default values as defined in the "train" entrypoint of the ``MLproject`` file.

* **Valid run example**: <br>
``mlflow run --experiment-name 'asset.variety' --entry-point gridsearch . -P task='asset.variety' -P metric='f1' -P averaging='macro' -P imputer='dropnan' -P random_state=0 -P merge='yes' -P pca=0 -P n_folds=5 -P n_jobs_cv=-1  --no-conda``

The MLflow UI (http://localhost:5000) can be used for tracking model performance metrics. Models, artifacts and tracking parameters are stored locally.

Deployment with docker
----------------------

* **Prerequisites**: Docker, docker-compose

* **Description**: <br>
This is the native application of RiskML. RiskML contributes to the Cyber Situational Awareness of a system by introducing knowledge and intelligence from external data sources (VCDB). RiskML maintains an updated version of VCDB. Based on different feature subsets (different input / target combinations) of the VCDB dataset, long-term estimations of the following threat related likelihoods are provided using appropriately trained Machine Learning algorithms from the python [sklearn](https://scikit-learn.org/stable/) library:

  * **Likelihood of occurrence of cyberthreats** such as different types of hacking and malware, social engineering, and human errors: <br><br>
P ( threat type |  organisation industry,organisation size,asset type) <br><br>
The occurrence of each threat type has been modelled by a unique ML model that produces the corresponding predictions during the normal functioning of SPHINX. This estimation contributes to the overall awareness by predicting the level of exposure/sensitivity of each SPHINX network asset (e.g. database server, laptop/desktop end-user device) to the different cyber threats (e.g. rootkit, ransomware, DoS). 

  * **Likelihood of an asset to be victim of a particular threat** that the system has already been exposed to and has been reported by the SIEM: <br><br>
P ( asset type |  organisation industry,organisation size,threat type) <br><br>
Different machine learning models have been trained for each one of the asset types. These estimations contribute to the overall awareness by estimating the probability of a specific asset to be targeted by a threat that is already apparent in the system. 

  * **Likelihood of an incident to affect Confidentiality, Integrity, Availability** i.e. each one of the attributes of the Information Security Triad. <br><br>
P ( attribute |  organisation industry,organisation size,threat type,asset type) <br><br>
This estimation contributes to the overall awareness by estimating the probability of an attack that exhibits specific characteristics to target each one of the CIA triad. One model is trained for each one of the three attributes returning some risk predictions that depend on the nature of the incident based on past incidents from the real world. For example, the model will return a high-risk value for Availability when the threat type is a DDoS attack, accompanied by low values for Confidentiality and Integrity. <br><br>
For each one of these estimations different training tasks are defined. For each task, different machine learning algorithms (i.e. K-nearest neighbours (KNN), LightGBM (LGBM), Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR) are trained (algorithm choice configurable) and the one with the best performance is selected and deployed for inference. The VCDB dataset can be updated, and the models can be re-trained on-demand (http://localhost:5100/api/ui), thus providing more up-to-date and accurate threat occurrence likelihood estimations. Models are served for inference and can be consumed from other agents / end-users of the environment.

* **Configuration**: <br>
Deployment settings can be configured before running the deployment in the ``deployment_config.json`` file in json format. However **each one of the different trained models (different combinations of <task>.<target> & algorithm) gets its configuration (default arguments) from the "train" entrypoint of ``MLproject`` file except for task, target and algo which are explicitly set during the deployment phase, thus overriding the MLproject defaults**. Algorithms to be trained, logged and compared can be chosen along with the models to be trained and the respective ports to serve them for inference. The ``shell_scripts/shell_script_composer.py`` file is employed at the begining of the deployment, producing the appropriate shell scripts to handle th MLflow cli (see ``shell_scripts/discover`` and ``shell_scripts/kill`` folder).

* **Run**: <br>
Use docker-compose to deploy all services. In the root directory run:

``docker-compose build``

``docker-compose up`` 

The deployed component architecture is the following:

![RiskML_component_diagram](https://github.com/pelekhs/RiskML/blob/dev/report/images/RiskML_component_diagram.png)

After the deployment the following UIs can be used:

* Mlflow tracking server: <br>
http://localhost:5000
Users can monitor model performance, registered models and saved artifacts.

* MINIO model storage: <br>
http://localhost:9000 <br>
username: minio-id <br>
pass: minio-key
Users can inspect the model storage file system via a Web UI.

* Administrative interface of the deployment (Permits to check health of served models, update VCDB & retrain models): <br>
http://localhost:5100/api/ui
Operations of this API / UI are handled via shell scripts (``shell_scripts/discover`` and ``shell_scripts/kill`` folder) that are triggered by functions developed in the ``api/functions.py`` file. Shell scripts  are created by the ``shell_scripts/shell_script_composer.py`` file as mentioned above depending on the ``deployment_config.json`` configuration file.

* Served models API endpoints: <br>
http://localhost:<model_port>/invocations <br>
Valid API call: ``curl http://0.0.0.0:5011/invocations -H 'Content-Type: application/json' -d @input_examples/asset.variety.json``
