#!/bin/bash       


source /pbs/home/b/barcelin/pbs_throng_link/lsst_stack/loadLSST.bash

###setup lsst_distrib

cd /pbs/home/b/barcelin/pbs_throng_link/parameter_estimation/scripts/
sudo setup lsst_distrib
python generate_dc2_img.py
