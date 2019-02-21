#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:10:57 2018

@author: frederik
"""

import sys
from scipy.linalg import *
from scipy import *
import scipy.sparse as sp
from scipy.sparse import *
from numpy.random import *
from time import *
import os as os 
import os.path as op
from scipy.misc import factorial as factorial
import datetime
import logging as l
#from numpy import ufunc as ufunc
import TightBindingModel as TBM 

import time
#import scipy.ufunc as ufunc

Module=sys.modules[__name__]

SX = array(([[0,1],[1,0]]),dtype=complex)
SY = array(([[0,-1j],[1j,0]]),dtype=complex)
SZ = array(([[1,0],[0,-1]]),dtype=complex)
I2 = array(([[1,0],[0,1]]),dtype=complex)
#
#Dimension = 3
#OribtalDimension= 2 




def CheckEnvironmentVars():
    if not ("Dimension" in globals() and "OrbitalDimension" in globals()):
        raise NameError("Dimension and orbital dimension must be set. \n Specify TBM. Dimension and TBM.OrbitalDimension before declaring variables")

#def CheckLattice():
#    if not "Lattice" in globals():
#        raise NameError("Lattice object must be specified")
#    elif not type(Lattice)==Lattice:
#        print(type(Lattice))
#        raise NameError("Lattice must be a lattice object.")
#        
#    
        
class Lattice:
    """Lattice object -- contains all lattice-specific functions and objects
    Can be combined with Clean Hamiltonian to generate a Hamiltonian matrix"""
    def __init__(self,LatticeDimension,PBC=False):
        
        CheckEnvironmentVars()
#        
#        if type(LatticeDimension)==int:
#            LatticeDimension=tuple([LatticeDimension])
#        
#        # Checking for right input
#        if not shape(LatticeDimension)[0]==Dimension:
#            raise ValueError("Dimension of lattice sites must match dimension of syste")
#                    
        self.OrbitalDimension=OrbitalDimension
        self.PBC=PBC
        self.Dimension=Dimension
        self.Hdimension=self.OrbitalDimension*prod(LatticeDimension)
        
        if Dimension>1:            
            self.LatticeDimension=LatticeDimension
        else:
            self.LatticeDimension=(LatticeDimension,)
            
        self.UnitCells=prod(LatticeDimension)
        
        self.Ilist=self.gen_Ilist()
        self.Tlist=self.gen_Tlist()
#        
#        global Hdimension 
#        
        Module.Hdimension = self.Hdimension
        Module.LatticeDimension = LatticeDimension
        Module.OrbitalDimension = OrbitalDimension
        Module.PBC = PBC
    
    def get_index(self,Coords,Orbitals):
        if amax(Orbitals) >= self.OrbitalDimension:
            raise ValueError("Orbital index exceeds orbital dimension")
            
        Index = Orbitals
        Q=self.OrbitalDimension
        
        for d in range(0,self.Dimension):
            Index = Index + Coords[d]*Q    
            Q=Q*self.LatticeSize[d]
        
        return Index 
    
    def gen_Tlist(self   ):
        Tlist=[]
        for d in range(0,self.Dimension):
            D=self.LatticeDimension[d]
            
            
            Data=ones(D)

            if not self.PBC:
                Data[D-1]=0
                
                
            Cols=arange(D)
            Rows=(Cols+1)%D
    # 
            T=csr_matrix(coo_matrix((Data,(Rows,Cols)),dtype=complex))
            
            Tlist.append(T)
        
        return Tlist

    def gen_Ilist(self):
        Ilist=[] #csr_matrix(eye(self.OrbitalDimension),dtype=complex)]
        
        for d in range(0,self.Dimension):
            D=self.LatticeDimension[d]
            Ilist.append(csr_matrix(eye(D),dtype=complex))
        return Ilist
    
    def Identity(self,format="csr"):
        I = eye(self.Hdimension,format=format)
        return I
    
    def CoordinateVec(self,Dim):
        if Dim + 1 > self.Dimension:
            raise IndexError("Direction argument must match the lattice dimension")
            
        LatticeIndices = arange(0,self.Hdimension)//self.OrbitalDimension
        Vec=(LatticeIndices//prod(self.LatticeDimension[:Dim]))%self.LatticeDimension[Dim]
        
        return Vec
    
    def CoordinateOperator(self,Dim,format="csr"):
        CoordinateVec=self.CoordinateVec(Dim)
        
        Mat = eye(self.Hdimension,format=format)
        Mat.setdiag(CoordinateVec)
        return Mat
    
        

    
def gen_OnSitePotential(Potential):
    CheckLattice()
    
    """ 
    Generates a Hamiltonian corresponding to on-site potential
    
    The on-site potential A must be an ndarray, where A[nx,ny,..] indicates the on-site potential on site [nx,ny,...]
    """
    
    # Checking for right input"
    
    if not type (Potential)==ndarray or shape(Potential)!=tuple(Lattice.LatticeDimension):
        print(shape(Potential))
        print(Lattice.LatticeDimension)
        raise TypeError("Shape of potential must be ndarray and match physical Hamiltonian")
        
    Vec = ravel(Potential,order="F")
    
    Data=Vec
    Rows=arange(0,Lattice.UnitCells)
    Cols=arange(0,Lattice.UnitCells)
    LatticeMatrix = csr_matrix((Data,(Rows,Cols)),dtype=complex)

    I = csr_matrix(eye(Lattice.OrbitalDimension),dtype=complex)

    return kron(LatticeMatrix,I,format="csr")    
    

def get_density(Vec):
    CheckLattice()
    
    Rho=abs(Vec)**2
    
    Shape=tuple([OrbitalDimension]+list(Lattice.LatticeDimension)+[shape(Rho)[1]])
    RhoMat=reshape(Rho,Shape,order="F")
    
    RhoLattice = sum(RhoMat,axis=0)
    return RhoLattice

    
    
def VecToMatrix(Vec):
    Shape=tuple([OrbitalDimension]+list(Lattice.LatticeDimension))
    
    return reshape(Vec,Shape,order="F")

def MatrixToVec(Mat):
    return ravel(Mat,order="F")    

def get_Identity(format="csr"):
    
   return eye(Hdimension,format=format,dtype=complex)
    
def get_coords(index):
    
    orbital = index%OrbitalDimension
    latticeindex=index//OrbitalDimension
    
    coordlist=[]
    for d in range(0,Dimension):
        coord=(latticeindex//prod(LatticeSize[:d]))%LatticeSize[d]
        coord=cottord.astype(int)
        
        coordlist.append(coord)
        
    return [coordlist,orbital]


def LatticeHamiltonian(Hamiltonian,Lattice,format="csr"):
    """ Generates real space Hamiltonian matrix corresponding to lattice. 
    orbital d, position r=[x,y,z] corresponds to index d+x D_orbital + y D_orbital Lx + z D_orbital Lx Ly"""
#    CheckLattice()
    
    if not type(Hamiltonian)==TBM.Hamiltonian:
        raise ValueError("Hamiltonian must be BZO Hamiltonian")
    
    Hdimension = Lattice.Hdimension
    Dimension=Lattice.Dimension   
    Tlist=Lattice.Tlist
    
    NNZ=Hamiltonian.NNZ()
    IndList=Hamiltonian.IndList()
    
    Matrix = csr_matrix((Hdimension,Hdimension),dtype=complex).asformat(format)
    for Ind in IndList:
        OrbitalMat = Hamiltonian[Ind]

        Out = csr_matrix(OrbitalMat,dtype=complex)

        for d in range(0,Lattice.Dimension):
            if Ind[d]>=0:
                power = Ind[d]
                Mat = Tlist[d]
            if Ind[d]<0:
                power = - Ind[d]
                Mat = Tlist[d].conj().T
                
            t = pow(Mat,power)
            

            
            Out = kron(t,Out,format=format)

        


        Matrix = Matrix + Out
    
    return Matrix



