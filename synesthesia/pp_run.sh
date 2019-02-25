#!/usr/bin/env bash
#
# LPM | Author: Ben Marten
# Copyright (c) 2017 Leanplum Inc. All rights reserved.

paperspace jobs create \
--machineType K80 \
--isPreemptible true \
--container e7mac/tensorflow-gpu \
--workspace https://github.com/e7mac/ml-art \
--command cd synesthesia && python train.py \
--output=/storage \
--ports 6006:6006