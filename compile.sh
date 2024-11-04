#export PYTHONPATH=/home/comp/15485625/.local/lib64/python3.6/site-packages
rm -rf tcmm_cpp.egg-info
rm -rf build 
python3 setup.py clean 
#python3 setup.py install --prefix=/home/comp/15485625/.local
python3 setup.py install --prefix=/opt/conda/envs/test
# python3 tests/compress.py
