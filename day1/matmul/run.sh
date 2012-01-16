#! /bin/bash

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CONFIG=cuda_profile.cfg


export COMPUTE_PROFILE_LOG=profile.matmul1.log 
./matmul1 2048

export COMPUTE_PROFILE_LOG=profile.matmul2.log 
./matmul2 2048
