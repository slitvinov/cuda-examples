#! /bin/bash

set -e
set -u
export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CONFIG=cuda_profile.cfg
export COMPUTE_PROFILE_CSV=1

function prof() {
    local bin=$1
    for n in ${nlinst}; do
	local pfile=profile.${bin}.n${n}.log
	export COMPUTE_PROFILE_LOG=${pfile}
	./${bin} ${n} >/dev/null
	
	local gtime=$(grep "${bin}" ${pfile} | awk -v FS=","  '{print $2}')
	
	echo ${n} ${gtime}
    done > ${bin}.dat

}

nlinst="16 32 64 128 256 512 1024 2048 4096"
prof matmul1
prof matmul2


