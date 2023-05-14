#!/bin/bash

# A script to submit an experiment to the cluster.
# Usage: ./submit.sh [experiment|jupyter|sync] [sync-large-files|skip-large-files]

# Verify that the first argument is either `experiment`, `jupyter` or `sync`.
if [[ ! $1 =~ ^(experiment|jupyter|sync)$ ]]; then
  echo "Usage: $0 [experiment|jupyter|sync] [sync-large-files|skip-large-files]"
  exit 1
fi

_REMOTE=z1158649@gw.gmum  # The remote machine.
_WHAT_TO=${1-"experiment"}  # experiment or jupyter or sync
_SYNC_ALL_FILES=${2-"sync-large-files"}  # sync-large-files or skip-large-files

# Create a `vqa` directory on the remote machine if it doesn't exist.
ssh -T $_REMOTE << 'EOL'
  mkdir -p $HOME/vqa
EOL

# Quit if .env.prod doesn't exist.
if [[ ! -f .env.prod ]]; then
  echo "File .env.prod doesn't exist"
  exit 1
fi

# Copy the .env.prod file to the remote machine.
scp .env.prod $_REMOTE:./vqa/.env.prod

# Copy the code to the remote machine.
if [[ "$_SYNC_ALL_FILES" == "sync-large-files" ]]; then
  echo "Syncing all the files"
  rsync -vrzhe ssh ./ $_REMOTE:./vqa
elif [ "$_SYNC_ALL_FILES" == "skip-large-files" ]; then
  echo "Skipping large files"
  rsync -vrzhe ssh --exclude 'data' --exclude '.git' --exclude '.dvc' --max-size=10M ./ $_REMOTE:./vqa
else
  echo "Usage: $0 [experiment|jupyter|sync] [sync-large-files|skip-large-files]"
  exit 1
fi

# Run the experiment on the remote machine.
if [[ $_WHAT_TO == "experiment" ]]; then
   echo "Submitting experiment"
   ssh -T $_REMOTE << 'EOL'
    cd $HOME/vqa \
    && cp .env.prod .env \
    && source .env \
    && cd scripts \
    && sbatch run-experiment.sh \
    && echo "Experiment submitted" \
    && squeue -u "$USER"
EOL
elif [[ $_WHAT_TO == "jupyter" ]]; then
    echo "Submitting jupyter"
    ssh -T $_REMOTE << 'EOL'
      cd $HOME/vqa \
      && cp .env.prod .env \
      && source .env \
      && cd scripts \
      && sbatch run-jupyter.sh \
      && echo "Jupyter submitted" \
      && squeue -u "$USER"
EOL
elif [[ $_WHAT_TO == "sync" ]]; then
    echo "Copying the .env.prod file to .env"
    ssh -T $_REMOTE << 'EOL'
      cd $HOME/vqa \
      && cp .env.prod .env
EOL
else
  echo "Usage: $0 [experiment|jupyter|sync]"
  exit 1
fi
