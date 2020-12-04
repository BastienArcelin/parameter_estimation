#!/bin/bash  

#$ -P P_lsst

#$ -l os=cl7 
#$ -l sps=1
#$ -l s_fsize=4G
#$ -l s_cpu=02:00:00
#$ -l s_rss=10G 

#$ -M arcelin@apc.in2p3.fr
#$ -m be   ## envoie un email quand le job commence et termine 

#$ -o /sps/lsst/users/barcelin/job_outputs/dc2/ 
#$ -e /sps/lsst/users/barcelin/job_outputs/dc2/

source /pbs/home/b/barcelin/pbs_throng_link/lsst_stack/loadLSST.bash
setup lsst_distrib

cd /pbs/home/b/barcelin/pbs_throng_link/parameter_estimation/scripts/

##python generate_dc2_img.py 3262 validation_mag_24.5 10000 ## Vadidation
##python generate_dc2_img.py 4855 test_mag_26.5 100 ## Test
##python generate_dc2_img.py 4855 blend 100 ## Test 

python generate_dc2_img.py 4438 training_mag_26.5 10000 ## Training 5 in 4438 and 3 in 3261
