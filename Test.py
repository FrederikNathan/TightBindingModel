#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:34:58 2019

@author: frederik

Testing module
"""

import TightBindingModel as TBM
TBM.DIMENSION=1


#BZO = TBM.BZO


#TightBindingModel.DIMENSION = 2

A=TBM.BZO(shape=(1,))

A[1]=array([2])

A[2]=(3,)


#A[2,]=4
#A[(5,)]=87