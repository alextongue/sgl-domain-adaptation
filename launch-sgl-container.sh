#!/bin/sh

set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_NUM_GPU=1  # max of 2 (contact ETS to raise limit)
K8S_NUM_CPU=4  # max of 8 ("")
K8S_GB_MEM=32  # max of 64 ("")
K8S_TIMEOUT_SECONDS=(3600*24) # note: be careful with this! 48h timeout should be max.


# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=YES

# Sets up proxy URL for Jupyter notebook inside 
#K8S_INIT="/opt/k8s-support/bin/pause"
#K8S_ENTRYPOINT_ARGS=()
#SPAWN_INTERACTIVE_SHELL=NO

PROXY_ENABLED=YES
#K8S_ENTRYPOINT="/bin/sleep"
#K8S_ENTRYPOINT_ARGS_EXPANDED='"8640000"'

K8S_DOCKER_IMAGE="ucsdets/scipy-ml-notebook:2020.2.9"
#K8S_ENTRYPOINT="~/sgl-domain-adaptation/run_jupyter_sgl.sh"

exec /software/common64/dsmlp/bin/launch.sh -i alextongue/sgl-domain-adaptation -P Always "$@"

