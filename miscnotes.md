#### Installing FAISS

###### Install CMake

Starting from your home directory,

1. `sudo apt install gcc`
	* Answer `Y`
	* This installs gcc, a C compiler. Necessary to bootstrap CMake.

2. `sudo apt install g++`
	* Answer `Y`
	* This installs g++, a C++ compiler. Necessary to bootstrap CMake.

3. `sudo apt-get install build-essential`
	* Answer `Y`
	* This installs a Makefile processor. Necessary to bootstrap CMake.

4. `sudo apt-get install libssl-dev`
	* Necessary to bootstrap CMake.

5. `wget https://github.com/Kitware/CMake/releases/download/v3.20.0-rc2/cmake-3.20.0-rc2.tar.gz`

6. `tar -xzvf cmake-3.20.0-rc2.tar.gz`

7. `cd cmake-3.20.0-rc2`

8. `./bootstrap`

9. `make`

###### Installing swig

This is yet another necessary step before we can actually install FAISS. Starting from your home directory,

1. `sudo apt-get install libpcre3 libpcre3-dev`
	* Answer `Y`

2. I believe this gets installed in your miniconda3 or it comes with it (idk). Either way, in your `.bashrc`, add the line `export LD_LIBRARY_PATH=/home/<username>/miniconda3/pkgs/pcre-8.44-he6710b0_0/lib:$LD_LIBRARY_PATH`. Replace <username> with your username.

3. `source .bashrc`


4. `wget https://sourceforge.net/projects/swig/files/swig/swig-4.0.2/swig-4.0.2.tar.gz`

5. `tar -xzvf swig-4.0.2.tar.gz`

6. `cd swig-4.0.2.tar.gz`

7. Replace <username> with your username: `./configure --prefix=/home/<username>/`

8. `sudo make`

9. `sudo make install`

10. Check to see if it's working by running `swig -version`. You should get no error messages.


###### Actually installing FAISS

Starting from your home directory,

1. `git clone https://github.com/facebookresearch/faiss.git faiss_xx`

2. `cd faiss_xx`

3. `target_dir=$PWD/install_py`

4. `cmake=~/cmake-3.20.0-rc2/bin/cmake`

5. Replace <username> with your username: `$cmake -B build     -DFAISS_ENABLE_GPU=OFF     -DBLA_VENDOR=Intel10_64_dyn     -DMKL_LIBRARIES=$CONDA_PREFIX/lib     -DPython_EXECUTABLE=$(/home/<username>/miniconda3/bin/)     -DFAISS_OPT_LEVEL=avx2     -DCMAKE_BUILD_TYPE=Release`

6. `make -C build -j 10`


