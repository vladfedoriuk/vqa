#!/bin/bash

# A script to submit an experiment to the cluster.
_REMOTE=z1158649@gw.gmum

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
rsync -vrzhe ssh ./ $_REMOTE:./vqa

# Run the experiment on the remote machine.
if [[ $1 == "experiment" ]]; then
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
elif [[ $1 == "jupyter" ]]; then
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
elif [[ $1 == "sync" ]]; then
    echo "Copying the .env.prod file to .env"
    ssh -T $_REMOTE << 'EOL'
      cd $HOME/vqa \
      && cp .env.prod .env
EOL
else
  echo "Usage: $0 [experiment|jupyter]"
  exit 1
fi
