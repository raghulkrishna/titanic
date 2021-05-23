#!/usr/bin/env bash


for i in *.png *.jpg *.jpeg;
do
  echo "$i"
  cml-publish "$i" --md >> report.md
done