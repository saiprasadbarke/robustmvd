#!/bin/bash

if [ -z "$1" ]
then
   echo "Please path a target path to this script, e.g.: /setup_vismvsnet.sh /path/to/vismvsnet.";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading Vis-MVSNet repository https://github.com/jzhangbs/Vis-MVSNet.git to $TARGET."
mkdir -p "$1"

git clone https://github.com/jzhangbs/Vis-MVSNet.git $TARGET

OLD_PWD="$PWD"
cd $TARGET