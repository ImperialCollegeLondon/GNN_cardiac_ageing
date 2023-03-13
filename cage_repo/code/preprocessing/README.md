# Preprocessing

Docker build the following, with name mirtk_with_pandas (docker build . -t mirtk_with_pandas)

    FROM biomedia/mirtk
    ARG DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && apt-get upgrade
    RUN apt-get install -y python3-pandas python3-tqdm python3-xlrd python3-pip
    RUN pip3 install vtk ipython ipdb

then cd to repository main dir and run

<code>sudo docker run -it --rm -w '/4ds' -v \`pwd\`:/4ds:Z -v $HOME/cardiac:/cardiac:Z mirtk_with_pandas /bin/bash -c 'PYTHONPATH=/usr/local/lib/mirtk/python ipython code/preprocessing/data_preprocessing_1.py --pdb -- --subject /cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB'</code>

<code>sudo docker run -it --rm -w '/4ds' -v \`pwd\`:/4ds:Z -v $HOME/cardiac:/cardiac:Z mirtk_with_pandas /bin/bash -c 'PYTHONPATH=/usr/local/lib/mirtk/python ipython code/preprocessing/data_preprocessing_2.py --pdb -- --mesh /cardiac/minacio/LVendo_ED_template.vtk'</code>

