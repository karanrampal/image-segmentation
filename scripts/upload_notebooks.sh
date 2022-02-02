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

commit="$(git rev-parse HEAD)"

# Upload wheel file
echo "Upload wheel file"
bash scripts/upload_whl.sh "$branch" "$user" "$profile"
temp_whl_file="$(find dist/ImageSegmentation*whl | tail -n 1)"
whl_file="$(basename "$temp_whl_file")"

# Adjust notebook variables
rm -rf /tmp/notebooks
cp -r ./notebooks /tmp/
echo "Update notebooks parameters"
for notebook in /tmp/notebooks/*.py; do
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e "s|dbutils.widgets.get(\"branch_name\")|\"${branch}\"|" "${notebook}"
    sed -i '' -e "s|dbutils.widgets.get(\"username\")|\"${user}\"|" "${notebook}"
    sed -i '' -e "s|dbutils.widgets.get(\"commit_id\")|\"${commit}\"|" "${notebook}"
    sed -i '' -e "s|# Install requirements command|%pip install -r /dbfs/imseg/${user}/${branch}/requirements.txt|" "${notebook}"
    sed -i '' -e "s|# Install wheel command|%pip install /dbfs/imseg/${user}/${branch}/$whl_file|" "${notebook}"
  else
    sed -i -e "s|dbutils.widgets.get(\"branch_name\")|\"${branch}\"|" "${notebook}"
    sed -i -e "s|dbutils.widgets.get(\"username\")|\"${user}\"|" "${notebook}"
    sed -i -e "s|dbutils.widgets.get(\"commit_id\")|\"${commit}\"|" "${notebook}"
    sed -i -e "s|# Install requirements command|%pip install -r /dbfs/imseg/${user}/${branch}/requirements.txt|" "${notebook}"
    sed -i -e "s|# Install wheel command|%pip install /dbfs/imseg/${user}/${branch}/$whl_file|" "${notebook}"
  fi
done

# Upload notebooks
# Please uncomment `--profile <profile name>` option if you have multiple databricks profile configured locally.
if [[ "$OSTYPE" == "msys"* ]]; then
  # Windows
  rm -rf temp_notebooks
  cp -r /tmp/notebooks temp_notebooks
  databricks workspace mkdirs "/imseg/${user}/${branch}" --profile "$profile"
  databricks workspace import_dir temp_notebooks "/imseg/${user}/${branch}" --overwrite --profile "$profile"
  rm -rf temp_notebooks
else
  databricks workspace mkdirs "/imseg/${user}/${branch}" --profile "$profile"
  databricks workspace import_dir /tmp/notebooks "/imseg/${user}/${branch}" --overwrite --profile "$profile"
fi

# Cleanup
rm -rf /tmp/notebooks

echo "Done"
