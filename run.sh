export PYTHONPATH=$PYTHONPATH:dist/Theano:dist/blocks
export THEANO_FLAGS='device=gpu,floatX=float32,optimizer=fast_run,lib.cnmem=0.9'
rsync ~/data/dogs_vs_cats.hdf5 $LSCRATCH
export FUEL_DATA_PATH=$LSCRATCH
python -u ~/dist/ift6266h16/main.py $@
