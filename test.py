#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:12:47 2019

@author: zhouge
"""

import json


with open('patch_results.json', 'w') as fp:
    data = json.load(fp)
#print(data[1])