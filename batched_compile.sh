#export PYTHONPATH=/home/comp/15485625/.local/lib64/python3.6/site-packages
rm -rf batched_tcmm_cpp.egg-info
rm -rf build 
python3 batched_setup.py clean 
#python3 setup.py install --prefix=/home/comp/15485625/.local
python3 batched_setup.py install --prefix=/opt/conda/envs/test
# python3 tests/batched_compress.py

