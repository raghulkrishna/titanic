#!/usr/bin/env bash




for file in *.{jpg,jpeg,png}; do
  [ -e "$file" ] || continue
  # Here "$file" exists
   echo "$file"
   cml-publish "$file" --md >> report.md
   echo "  " >> report.md

done