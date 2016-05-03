#!/bin/sh
if [ ! -d "build" ]; then
  ./buildit.sh
fi
build/bin/$1 obj/$2.obj
