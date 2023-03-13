import subprocess
import time
import itertools
import os
import numpy as np

# Script to run all models
# e.g.: run this file on terminal via:
# cp run_all.py run_all_tmp.py; screen -dm ipython run_all_tmp.py

sleep_time = .01
min_cuda_dev_id = 0
max_cuda_dev_id = 7
# Windows network folders workaround: avoids locking run.py file
subprocess.run('cp run.py run_tmp.py', shell=True)

product_list = [
   # ('lasso', 'lv_nlv_t1'), ('lasso', 'lv_nlv'),
   # ('catb', 'lv_nlv_t1'), ('catb', 'lv_nlv'),
   # ('nn', 'mesh'), ('nn', 'nlv_and_mesh'), ('nn', 'lv_nlv_t1_mesh'),
   # ('nn', 'mesh'), ('nn', 'nlv_and_mesh'), ('nn', 'lv_nlv_t1_mesh'),
   ('gnn', 'mesh'),
]*7

decimate_levels = itertools.cycle([.9, .93])
subsample_factors = itertools.cycle([.5, .25, .75])
subsample_factors = itertools.cycle([1.])


cuda_dev_id = min_cuda_dev_id-1
for esttype, modeln in product_list:
    if esttype in ['gnn', 'nn']:
        cuda_dev_id += 1

    screen_name = f'{cuda_dev_id} {modeln} {esttype}'

    cmd = 'nvidia-docker run -it --rm '
    cmd += '-v /dev/shm/:/dev/shm:Z -v $HOME/cardiac:/root/cardiac:Z -v $HOME/minacio:/root/minacio:Z '
    cmd += '-v $HOME/.cache:/root/.cache:Z -v /scratch/minacio/:/scratch/minacio/:Z -w /root/${PWD#"$HOME"/} '
    cmd += f'--memory=50g --cpus=15 -e CUDA_VISIBLE_DEVICES={cuda_dev_id} '
    cmd += f'meshtools ipython run_tmp.py -- --modeln={modeln} --esttype={esttype} --n_trials=100'
    if esttype == 'gnn':
        subsample_factor = next(subsample_factors)
        cmd += f' --subsample_factor={subsample_factor}'
        screen_name += f" sf{subsample_factor}"
        if subsample_factor == 1:
            decimate_level = next(decimate_levels)
            cmd += f' --gnn_decimate_level={decimate_level}'
            screen_name += f" dl{decimate_level}"

    if cuda_dev_id == max_cuda_dev_id:
        cuda_dev_id = min_cuda_dev_id-1
        break
    print(cmd)
    subprocess.run(
        f"screen -S '{screen_name}' -dm bash -c 'for i in `seq 1 1000`; do sleep 1 && ({cmd} || (sleep $((5*($i))) && bash -c \"exit 1\")) && break; done; sleep 3600'",
        shell=True)
    time.sleep(sleep_time)
