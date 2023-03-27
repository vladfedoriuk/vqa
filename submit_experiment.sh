#!/bin/bash

# A script to submit an experiment to the cluster.
_REMOTE=z1158649@gw.gmum

# Copy the code to the remote machine.
# This will omit some large files - if you need them, rsync them manually:
#  - data/ - the datasets
#  - .dvc/ - the dvc cache
#  - .git/ - the git repo
rsync -vrzhe ssh ./ $_REMOTE:./vqa --exclude .dvc --exclude data --exclude .git
ssh -T $_REMOTE << 'EOL'
  cd vqa \
  && cp .env.prod .env \
  && export "$(grep -v '^#' .env | xargs -0)" \
  && sbatch scripts/run-experiment.sh \
  && echo "Experiment submitted" \
  && squeue -u "$USER"
EOL
