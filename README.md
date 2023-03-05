# VQA Experiments
A collection of experiments and explorations related to Visual Question Answering

## Local setup
### Create a virtual environment
There are many ways to create a virtual environment. Here is one of them using `pyenv`:
- If you don't have Python 3.10.10 installed, do it as following:
```shell
pyenv install 3.10.10
```
- Create a virtual environment:
```shell
pyenv virtualenv 3.10.10 vqa
```
- Activate it for the working directory:
```shell
pyenv local vqa
```
### Install dependencies and pre-commit hooks
```shell
make init-dev
```
or just
```shell
make
```
### Specify environment variables
```shell
cp .env.template .env
```
- Replace the `...` with corresponding values.
- To export environment variables one can use a command like this:
```shell
export $(grep -v '^#' .env | xargs -0)
```
### Working on server
- To copy a directory from local host to a remote host via SCP (example):
```shell
scp -r /local/directory/ username@to_host:/remote/directory/
```
- To copy a directory from a remote host to local host via SCP (example):
```shell
scp -r username@from_host:/remote/directory/ /local/directory/
```
#### Using Jupiter Notebook
- To run a Jupiter Notebook on the server:
```shell
export SINGULARITYENV_WANDB_API_KEY=<your-value>
export SINGULARITYENV_WANDB_ENTITY=<your-entity-name>
export HF_DATASETS_CACHE="/shared/sets/datasets/huggingface_cache"
export DATASETS_PATH="/shared/sets/datasets"
export SAVE_ARTIFACTS_PATH="/.local/share"
cd scripts && sbatch run-jupyter.sh
```
- To access the notebook, one can use a command like this:
```shell
ssh -N -f -L localhost:8989:localhost:8989 [username]@[server-job-runs-on].gmum
```
- To get the name of the server job runs on, one can use a command like this:
```shell
squeue -u [username]
```
The result should look like this:
```shell
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            123456     batch run-jup username  R       0:01      1 node-1
```
Then, one can use the `node-1` name to access the notebook.
- Then, one can access the notebook via a browser at `localhost:8989`.
- To stop the notebook, one can use a command like this:
```shell
scancel <job_id>
```
- To get the job id, one can use a command like this:
```shell
squeue -u <username> -n jupyter
```
- To get the list of all the running jobs, one can use a command like this:
```shell
squeue -u <username>
```
