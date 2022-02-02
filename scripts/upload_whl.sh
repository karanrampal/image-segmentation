#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cd ..

if [ $# == 0 ]; then
  branch=$(git rev-parse --abbrev-ref HEAD)
  user=$(whoami)
  profile="DEFAULT"
else
  branch=$1
  user=$2
  profile=$3
fi

# Build and upload package
echo "Generate whl-file"
rm -rf dist
#python setup.py sdist bdist_wheel
python -m build

temp_whl_file="$(find dist/ImageSegmentation*whl | tail -n 1)"
whl_file="$(basename "$temp_whl_file")"
echo "The wheel file is: ${whl_file}"
echo "Copy the wheel file."
dbfs cp dist/"${whl_file}" "dbfs:/imseg/${user}/${branch}/${whl_file}" --overwrite --profile "$profile"

echo "Copy requirements.txt"
dbfs cp -r requirements "dbfs:/imseg/${user}/${branch}/requirements" --overwrite --profile "$profile"

bash ./scripts/upload_configs.sh "$branch" "$user" "$profile"

echo "Done uploading wheel file"
