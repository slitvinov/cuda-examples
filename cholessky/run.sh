#! /bin/bash

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CONFIG=cuda_profile.cfg
export COMPUTE_PROFILE_CSV=1

n=10
pfile=profile.n${n}.log
export COMPUTE_PROFILE_LOG=${pfile}
./cholessky 100
