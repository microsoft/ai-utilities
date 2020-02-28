#!/usr/bin/env bash
git checkout master

var=$(sed -ne "s/version=['\"]\([^'\"]*\)['\"] *,.*/\1/p" ./setup.py)
IFS='.' read -r -a array <<<"$var"

major="${array[1]}"
minor="${array[2]}"

if [ "${array[2]}" -eq 9 ]; then
  echo $major
  major=$((major + 1))
  echo $major
else
  minor=$((minor + 1))
fi

version=0.$major.$minor

sed -i "s/version=['\"]\([^'\"]*\)['\"] *,.*/version=\"$version\",/" ./setup.py

git add setup.py
git commit -m "update version number for next release"
git push