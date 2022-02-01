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

commit="$(git rev-parse HEAD)"

# Upload wheel file
echo "Upload wheel file"
bash scripts/upload_whl.sh "$branch" "$user" "$profile"
temp_whl_file="$(find dist/hm2vec*whl | tail -n 1)"
whl_file="$(basename "$temp_whl_file")"

# Adjust notebook variables
rm -rf /tmp/notebooks
cp -r ./notebooks /tmp/
echo "Update notebooks parameters"
for notebook in /tmp/notebooks/*.py; do
  for device in cpu gpu; do
    mkdir -p "/tmp/notebooks/${device}"

    tmp_notebook="${notebook##*/}"
    device_notebook="/tmp/notebooks/${device}/${tmp_notebook}"
    cp "${notebook}" "${device_notebook}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' -e "s|dbutils.widgets.get('branch_name')|'${branch}'|" "${device_notebook}"
      sed -i '' -e "s|dbutils.widgets.get('username')|'${user}'|" "${device_notebook}"
      sed -i '' -e "s|dbutils.widgets.get('commit_id')|'${commit}'|" "${device_notebook}"
      sed -i '' -e "s|#Install requirements command|%pip install -r /dbfs/hm2vec/${user}/${branch}/requirements/requirements_dbx_${device}.txt|" "${device_notebook}"
      sed -i '' -e "s|#Install wheel command|%pip install /dbfs/hm2vec/${user}/${branch}/$whl_file|" "${device_notebook}"

      sed -i '' -e "s|#run demo_notebook command|%run /hm2vec/${user}/${branch}/${device}/demo_notebook|" "${device_notebook}"
    else
      sed -i -e "s|dbutils.widgets.get('branch_name')|'${branch}'|" "${device_notebook}"
      sed -i -e "s|dbutils.widgets.get('username')|'${user}'|" "${device_notebook}"
      sed -i -e "s|dbutils.widgets.get('commit_id')|'${commit}'|" "${device_notebook}"
      sed -i -e "s|#Install requirements command|%pip install -r /dbfs/hm2vec/${user}/${branch}/requirements/requirements_dbx_${device}.txt|" "${device_notebook}"
      sed -i -e "s|#Install wheel command|%pip install /dbfs/hm2vec/${user}/${branch}/$whl_file|" "${device_notebook}"

      sed -i -e "s|#run demo_notebook command|%run /hm2vec/${user}/${branch}/${device}/demo_notebook|" "${device_notebook}"
    fi
  done
  rm "${notebook}"
done

# Upload notebooks
# Please uncomment `--profile <profile name>` option if you have multiple databricks profile configured locally.
if [[ "$OSTYPE" == "msys"* ]]; then
  rm -rf temp_notebooks
  cp -r /tmp/notebooks temp_notebooks
  databricks workspace mkdirs "/Shared/hm2vec/production" --profile "$profile"
  databricks workspace mkdirs "/Shared/hm2vec/end_to_end_tests" --profile "$profile"
  databricks workspace mkdirs "/hm2vec/${user}/${branch}" --profile "$profile"
  databricks workspace import_dir temp_notebooks "/hm2vec/${user}/${branch}" --overwrite --profile "$profile"
else
  databricks workspace mkdirs "/Shared/hm2vec/production" --profile "$profile"
  databricks workspace mkdirs "/Shared/hm2vec/end_to_end_tests" --profile "$profile"
  databricks workspace mkdirs "/hm2vec/${user}/${branch}" --profile "$profile"
  databricks workspace import_dir /tmp/notebooks "/hm2vec/${user}/${branch}" --overwrite --profile "$profile"
fi

# Cleanup
rm -rf /tmp/notebooks

echo "Done"
