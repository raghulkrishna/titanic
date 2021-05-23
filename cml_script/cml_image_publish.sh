#!/usr/bin/env bash


for i in *.png *.bin *.txt;
do
  echo "$i"
  cml-publish "$i" --md >> report.md
done