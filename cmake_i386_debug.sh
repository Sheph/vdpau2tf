DIR_NAME=`basename ${PWD}`
mkdir -p ../${DIR_NAME}-debug
cd ../${DIR_NAME}-debug
cmake -G "Eclipse CDT4 - Unix Makefiles" \
-D_ECLIPSE_VERSION=4.3 \
-DCMAKE_BUILD_TYPE="Debug" \
-DTF_DIR=/opt/mnt/stas/1-tf/tensorflow \
../${DIR_NAME}
