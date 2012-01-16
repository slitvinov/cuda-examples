#! /bin/bash

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CONFIG=cuda_profile.cfg
export COMPUTE_PROFILE_CSV=1

nlinst="16 32 64 128 256 512 1024 2048"
for n in ${nlinst}; do
    export COMPUTE_PROFILE_LOG=profile.matmul1.n${n}.log 
    ./matmul1 ${n}
done

for n in ${nlinst}; do
    export COMPUTE_PROFILE_LOG=profile.matmul2.n${n}.log 
    ./matmul2 ${n}
done

