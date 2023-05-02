#!/bin/bash

# A script to submit an experiment to the cluster.
_REMOTE=z1158649@gw.gmum

# Create a `vqa` directory on the remote machine if it doesn't exist.
ssh -T $_REMOTE << 'EOL'
  mkdir -p $HOME/vqa
EOL
# Copy the code to the remote machine.
# This will omit some large files - if you need them, rsync them manually.
# The files that are omitted are:
# - some files ignored by git
# - the dvc cache
rsync -vrzhe ssh ./ $_REMOTE:./vqa

# Run the experiment on the remote machine.
if [[ $1 == "experiment" ]]; then
   echo "Submitting experiment"
   ssh -T $_REMOTE << 'EOL'
    cd $HOME/vqa \
    && cp .env.prod .env \
    && export "$(grep -v '^#' .env | xargs -0)" \
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
      && export "$(grep -v '^#' .env | xargs -0)" \
      && cd scripts \
      && sbatch run-jupyter.sh \
      && echo "Jupyter submitted" \
      && squeue -u "$USER"
EOL
else
  echo "Usage: $0 [experiment|jupyter]"
  exit 1
fi
