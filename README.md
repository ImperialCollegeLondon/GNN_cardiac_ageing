# Cardiac age prediction using graph neural networks for motion analysis

First, let's assume that your input mesh data is located at /mesh_data_source

## Preprocessing motion data into mirtk decimated meshes

This is preprocessing necessary to build the `cardiac_age_mesh_decimate_level.pkl` file used as datasource for the DNNs (covariates) and while not used as covariates data for GNN (GNN reads he vtks and decimate them individually on the fly while training), it is also used to generate the list of valid patient ids for the GNNs (because this preprocessing skips the patient without motion or with invalid motion).

The first step is to build the Docker image required for running the decimation process. So, first, if `mirtk_with_pandas` does not exist, create a Dockerfile and Docker build the following, with name `mirtk_with_pandas` (`docker build . -t mirtk_with_pandas`). The docker image tarball is also available at `~/cardiac/minacio/docker_images/mirtk_with_pandas.tar`.

    FROM biomedia/mirtk
    RUN apt-get update
    ARG DEBIAN_FRONTEND=noninteractive
    RUN apt-get install -y python3-pandas python3-tqdm python3-xlrd python3-pip
    RUN pip3 install vtk pyvista

After that, you can run the two stages of the preprocessing decimation process:

```bash
cd cage_repo && docker run -it --rm -w '/4ds' -v `pwd`:/4ds:Z mirtk_with_pandas /bin/bash -c 'PYTHONPATH=/usr/local/lib/mirtk/python ipython code/preprocessing/data_preprocessing_1.py --pdb -- --subject /mesh_data_source
```

```bash
cd cage_repo && docker run -it --rm -w '/4ds' -v `pwd`:/4ds:Z mirtk_with_pandas /bin/bash -c 'PYTHONPATH=/usr/local/lib/mirtk/python ipython code/preprocessing/data_preprocessing_2.py --pdb -- --mesh LVendo_ED_template.vtk --downsample-rate=97'
```

Note: for `LVendo_ED_template.vtk` above, you can obtain a decimation template at https://github.com/lisurui6/CMRSegment/tree/master/input/params or use an instance or average of your data a template.


## Running the models

The first step is to build the Docker image with required dependencies if not available already:

```bash
cd meshtools && docker build . -t meshtools
```
  
The docker image tarball is also available at `~/cardiac/minacio/docker_images/meshtools.tar`. 

Running the model for graph neural networks:

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools ipython --pdb run.py -- --modeln=mesh --esttype=gnn
```

Running the model for neural networks:

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools ipython --pdb run.py -- --modeln=mesh --esttype=nn
```

Running the model for catboost:

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools ipython --pdb run.py -- --modeln=lv --esttype=catb
```

Take a look at the `run.py` file for the full documentation of the command or run the `--help` argument to print on the screen:

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools python run.py --help
```

You can also use the `run_all.py` which you can configure to spawn multiple processes (for faster hyperparameter searching in a single model or running distinct model configuration in parallel) for distinct models.

```bash
cd cage_repo/code/models/ && python run_all.py
```

Note that you can start many processes at the same type for the same model or for distinct models, they will not conflict because `optuna` is prepared to handle that concurrency using atomic SQL datasets (in case SQLite, it even has sequential garanterees).

Once you fitted the model you can run the compare_estimators file to generate results comparing gnn, nn abd catboost:

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools ipython --pdb compare_estimators.py
```

And the `predict_age_delta.py` to generate age delta file that was mentioned before. This will also output the results of the linear regression of the age delta.

```bash
cd cage_repo/code/models/ && docker run -it --rm -v /dev/shm/:/dev/shm:Z -v $HOME/$USER:/root/$USER:Z -w /root/${PWD#"$HOME"/} -e CUDA_VISIBLE_DEVICES='1' meshtools ipython --pdb predict_age_delta.py
```

To generate interpretability results, run `gnn_interpretation_saliency.py` and `gnn_interpretation_explainer.py` in similar fashion as above (i.e.: inside Docker as above). Note that pytorch geometric is changing the interface for its interpretability explainer, so if in the future you want to run `gnn_interpretation_explainer.py`, then you must use `torch_geometric.__version__ == '2.0.4'` or will need to adapt the code to the new interface.

## Hairy hearts

To generate hairy hearts, first you need to install pyvista and paraview in the computer.
The recommended method is to install using mamba (or conda).
The best version available is Mambaforge from conda-forge team.

```code:: bash
  wget https://github.com/conda-forge/miniforge/releases/download/4.14.0-0/Mambaforge-Linux-x86_64.sh -o /tmp/mambaforge.sh
  chmod 770 /tmp/mambaforge.sh
  /tmp/mambaforge.sh
  mamba create --name paraviewenv paraview ipython pandas scipy python=3.10
```  
(Note: you can use conda instead of mamba in the step above, but it's slower).
Then you need to first generate average heart meshes for each age group.

```code:: bash
  cd cage_repo/code/hairy_hearts && ipython generate_files_average.py
```

Finally you can generate the hairy_hearts, open paraview, open its shell (toolbar View > Python Shell), then run

```code:: python
  import os
  os.chdir('cage_repo/code/hairy_hearts')
```  
  
and then click on "Run script" and select file `script_to_run_in_paraview_shell.py` in `cage_repo/code/hairy_hearts`

You can change the age group and baseline of relative motion by change the following line in the script_to_run_in_paraview_shell.py file:

```code:: python
  loop_coordinates_file_baseline='loop_coordinates_50_unhealthy.txt'
  loop_coordinates_file='loop_coordinates_70_unhealthy.txt'
```  
