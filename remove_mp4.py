# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:37:04 2020

@author: Poorna
"""

import glob,os
for i in glob.glob(os.path.join("C:/Users/Poorna/Desktop/photo","*.mp4")):
    try:
       os.chmod(i,0o777)
       os.remove(i)
    except OSError:
       pass
