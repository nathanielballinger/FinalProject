#!/bin/sh
if [ ! -d "build" ]; then
  mkdir build
fi
cd build
if [ "$1" = "debug" ]; then
  echo "Debug build with Clang Address Sanitizer."
  cmake -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O1 -std=c++11 -fsanitize=address \
    -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" ..
else
  echo "Release build."
  cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-std=c++11 -O3 " ..
fi
make
if [ "$1" = "debug" ]
then
  OS=`uname`
  if [ "$OS" = "Darwin" ]
  then
    echo "Running dsymutil."
    ls -1 bin | grep -v dSYM | xargs -I % dsymutil bin/%
  fi
fi
echo "Build finished."

