#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
cd ..

if [ $# == 0 ]; then
  branch=$(git rev-parse --abbrev-ref HEAD)
  user=$(whoami)
  profile="hm2vec"
else
  branch=$1
  user=$2
  profile=$3
fi

echo "Copy configs directory"
dbfs cp -r configs "dbfs:/hm2vec/${user}/${branch}/configs/" --overwrite --profile "$profile"

echo "Done"
