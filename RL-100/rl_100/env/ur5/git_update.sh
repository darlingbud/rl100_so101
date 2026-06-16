#!/bin/bash

DATE=$(date '+%Y-%m-%d %H:%M:%S')

if [ $# -eq 0 ]; then
  COMMIT_MESSAGE="$DATE - Update"
else
  COMMIT_MESSAGE="$DATE - $*"
fi

git add .
git commit -m "$COMMIT_MESSAGE"
git push

