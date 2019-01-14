pip install Pillow --user
pip install numpy --user
sudo apt-get install python-mpi4py
cd libs/nd2reader
python setup.py install --user
cd ../../openmp_extentions/
python install.py