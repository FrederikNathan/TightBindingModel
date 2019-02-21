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
from numpy import ufunc as ufunc

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

def CheckLattice():
    if not "Lattice" in globals():
        raise NameError("Lattice object must be specified")
    elif not type(Lattice)==LatticeObject:
        print(type(Lattice))
        raise NameError("Lattice must be a lattice object.")



class BZO:
    """
    BZO (Brillouin Zone Object):
    
    Syntax: BZO(*args,shape=None,dtype=complex) Fundamental object used to
    represent fields defined on a Brillouin zone. `.
    
    Any field on the BZ is a periodic function of crystal momentum, and
    can be decomposed into discrete harmonics (a Fourier series):
    
    F(k) = \sum_{abc} F_{abc} e^{-i (aq_1 +bq_2 + cq_3) \cdot k}. 
    
    In essence, the BZO represents a BZ field by storing a list of components
    {F_1 , \ldots F_n} (ObjList), along with their corresponding indices
    {(a_1,b_1,c_1),\ldots (a_n,b_n,c_n)} (IndList).
          
    The Harmonics \{F_{ijk}\} of a BZO can be accessed and set using the
    __set__ and __get__ method:
    
    BZO[i,j,k] =F_{ijk}.
    
    The value of F at a crystal momentum k (or a collection of crystal
    momenta) can be found using the __call__method :
    
    BZO(k) = F(k)
    
    see __call__ method for information on how multiple k-points should be 
    formatted
    
    Currently, TBM only works for square lattices: by default q_i is the ith
    unit vector. In a future version, the BZO can be updated to allow for
    general vectors.
            
    A BZO can be defined in 3 different ways
    
    Method 1: the BZO can be generated from an IndList and ObjList generated
    previously:
    
    A = <IndList> B = <ObjList> X = BZO(A,B)
    
    Method 2: the BZO from an (m_1 x ... m_n) array of BZO's
    
    A=<BZO_1> B=<BZO_2> C=array([A,B])
    
    X=BZO(C)
    
    Method 3: the BZO is defined as an empty object, and harmonics can be set
    afterwards. Here the shape should be specified (in the other two cases it
    should remain “None”).
    
    X=BZO(shape=(2,2),dtype=float) X[1,2,3]=array([[1,2],[3,4]])
    

    """


    def __init__(self,*args,shape=None,dtype=complex):

        # Check that Dimension and OrbitalDimension are set before doing anything
        CheckEnvironmentVars()

        ArgV=[arg for arg in args]

        # Determine which method is used to generate BZO:

        if len(ArgV)==2:
            # Method 1: Generate BZO from prefabrictated IndList and ObjList

            if not shape==None:
                raise ValueError("Shape cannot be defined, as it is set from the first argument")

            [IndList,ObjList]=ArgV

            # Set intrinsic parameters
            self.__Dimension = Dimension
            self.__shape=array(ObjList[0]).shape
            self.__dtype=ObjList[0].dtype

            # Define ZeroObject
            self.__Zero = self.__ZeroObject(self)

            # Set IndList and ObjList
            self.SetLists(IndList,ObjList)

        elif len(ArgV)==1:
            # Method 2: Generate BZO from an (m_1 x ... m_n) array of BZO's

            if not shape==None:
                raise ValueError("Shape cannot be defined, as it is set from the first argument")

            Obj = MergeBZOs(ArgV[0])

            # Set intrinsic parameters
            self.__shape=Obj.shape()
            self.__dtype=Obj.dtype()
            self.__Dimension = Dimension

            # Define ZeroObject
            self.__Zero = Obj.__Zero

            # Set IndList and ObjList
            self.SetLists(Obj.IndList(),Obj.ObjList())


        elif len(ArgV)==0:
            # Method 3: Create empty BZO

            if not (type(shape)==tuple or type(shape)==int):
                raise ValueError("Shape must be given by an integer or a tuple")

            #Set intrinsic parameters
            self.__Dimension=Dimension
            self.__shape=shape
            self.__dtype=dtype

            # Set lists (they are empty when using this method)
            self.SetLists([],[])

            # Define ZeroObject
            self.__Zero = self.__ZeroObject(self)

        else:
            # Otherwise raise error
            raise ValueError("BZO must be defined either from (IndList,ObjList), from array of BZOs or as an empty BZO")


    def SetLists(self,IndList,ObjList):
        """ Set IndList and ObjList for the BZO"""

        # Check that IndList and ObjList have the same length
        if not len(IndList)==len(ObjList):
            raise ValueError("IndList and ObjList must be of the same length")

        # Check that the indices are tuples of length self.__Dimension
        IndFormatError=0
        
        if not prod([type(x)==tuple for x in IndList]):
            IndFormatError=1

        elif not prod([len(x)==self.__Dimension for x in IndList]):
            IndFormatError=1

        if IndFormatError:
            raise ValueError(f"IndList must be list of tuples with length {self.__Dimension}")


        # Check that the shapes of the elements of ObjList are identical to self.__shape
        if not prod([shape(x) == self.__shape for x in ObjList])==1:
            print([shape(x) for x in ObjList])
            raise ValueError("ObjList must contain arrays of the same shape as the BZO (shape %s)"%str(self.__shape))

    
        # Set IndList and ObjList
        self.__IndList=IndList
        self.__ObjList=ObjList

        # Update sorting list (see __Set_NumList)
        self.__Set_NumList()
        
        # Sort lists
        self.__SortLists()


    def __Set_NumList(self,IndexCounter=None):
        """ 
        Generates NumList for BZO. The elements in IndList and ObjList are
        sorted according to their corresponding values in NumList. Here
        NumList[z] = self.__IndToNum(IndList[z]). The sorting makes the basic
        methods of BZO go much faster.
        
        The function __IndToNum takes as input the variable IndexCounter to
        convert Indices in IndList to integers. IndexCounter must always be
        larger than the larges integer in IndList (absolute value). It is by
        default 1e5, but if an integer in IndList exceeds this, IndexCounter
        is increased, and NumList are updated according to the new value of
        IndexCounter.
        """
        
        # Defualt value of Index counter
        DefaultIndexCounter=1e5

        # Determine IndexCounter
        if IndexCounter == None:
            
            # Determine indexcounter if not specified           
            if len(self.__IndList)==0:
                # Set to default if IndList and ObjList are empty
                IndexCounter = DefaultIndexCounter

            else:
                # Else, set to whichever is largest of the default value and the largest integer in IndList 
                
                IndexMax = max([max([abs(x) for x in Ind]) for Ind in self.__IndList])
                IndexCounter = int(max(DefaultIndexCounter,IndexMax)+0.1)

        elif IndexCounter > max([max([abs(x) for x in Ind]) for Ind in self.__IndList]):
            # Make sure that indexcounter is larger than the largest integer in IndList. Raise error if this is not the case 
            
            z = max([max([abs(x) for x in Ind]) for Ind in self.__IndList])
            raise ValueError(f"IndexCounter nust be larger than any integer in IndList ({z})")


        # Set IndexCounter        
        self.__IndexCounter=IndexCounter
        
        # Generate NumList
        self.__NumList=[self.__IndToNum(Ind) for Ind in self.__IndList]


    def __IndToNum(self,*Ind):
        """ 
        Method for converting index in IndList to integer. In 3d, the index
        (a,b,c) in IndList is converted to
        
        __IndToNum((a,b,c)) = a*k^2 + b*k^1 + c*k^0
        
        where k = self.__IndexCounter. A similar pattern works for other 
        dimensions.
        """

        return sum((Ind[0][n]*self.__IndexCounter**(self.__Dimension-n-1) for n in range(0,self.__Dimension)))


    def __CheckIndices(self,Index):
        # Checks that an index in IndList is a tuple of length self.__Dimension
        
        if not len(Index) == self.__Dimension:
            raise IndexError("Field is indiced by %d integers. Index given is %s"%(self.__Dimension,str(Index)))

    def __FindIndex(self,*Index):
        """ 
        Finds the list index z such that IndList[z] = Index. If Index is not
        in IndList, returns -1-z' such that X[z'+1] contains Index, where X is
        obtained by inserting Index in IndexList (while maintaining ordering
        wrt NumList)
        """

        # Check that index has right format
        self.__CheckIndices(*Index)

        # Find the integer corresponding to Index, using __IndToNum
        Num=self.__IndToNum(*Index)
        
        # Find the slot in NumList where Num would go (regardless of whether Num is in NumList or not)
        listindex= searchsorted(self.__NumList,Num)

        # Determine returned index:
        
        if listindex == self.NNZ():
            # If Num is larger than all elements in NumList, return -1-len(NumList) 
            return -1-listindex
        
        elif self.__NumList[listindex]==Num:
            # if Num is in NumList, return the index z where Num=NumList[z]
            
            return listindex

        else:
            # otherwise, return -1-z where z is the index where Num would be if it was inserted in NumList 
            
            return -1-listindex



    def __getitem__(self,*Index):
        """ 
        Return ObjList[z] where IndexList[z]=Index. If IndexList does not 
        contain Index, return self.__Zero (See below for definition). 
        Note that this method can be used to set elements of BZO, using 
        BZO[Index][a,b] = x,
        """
        
        # Find z where IndexList[z]=Index        
        n = self.__FindIndex(*Index)


        if n>=0 :
            # Return z if search was successful
            
            return self.__ObjList[n]

        else:
            # Otherwise, return zero object, with called index set to Index
            self.__Zero.CalledIndex=tuple(*Index)

            return self.__Zero

    
    class __ZeroObject:
        """
        Zero object: this object is returned if one used BZO[Ind]
        (BZO.__get__(Ind)) where Ind is not in IndList. The definition of
        ZeroObject allows to use BZO[Ind][a,b] in computations (as a matrix of
        zeros), and set the BZO with BZO[Ind][a,b] = x, even if Ind is not yet
        in IndList.
        
        ZeroObject is only used by BZO's __get__ method.
        """
        
        def __init__(self,bzo):
            """ 
            ZeroObject should always be defined with the host BZO as argument. 
            When called in arithmetic computations, it returns a matrix with 
            zeros in shape BZO.__shape
            """
            
            # Set intrinsic parameters           
            
            # Shape
            self.__shape=bzo.shape()
            
            # Matrix to be used for computations 
            self.__Mat = zeros(self.__shape,dtype=complex)
            
            # Host BZO
            self.__BZO = bzo
            
            # Called Index (Index where ZeroObject acts as a replacement for the zero matrix (needed when BZO[Ind]=x is used))
            # CalledIndex is set externally in BZO.__get__() before ZeroObject is used by this method (recall that this is the only place where ZeroObject is used).
            # Before first time of ZeroObject is used, CalledIndex is by default set to None 
            self.CalledIndex=None


        def __getitem__(self,Index):
            # Return zero matrxi when calling BZO[Ind]
             return self.__Mat[Index]*1

        def __setitem__(self,Index,Value):
            # Ensure that after BZO[Ind][1,1] = x, BZO[Ind] returns a matrix array([[0,0],[0,x]]) when BZO[Ind] was initially empty
            
            # Set matrix to enter in BZO's new ObjList
            NewMat = 1*self.__Mat
            
            NewMat[Index]=Value
            
            # Add new element to BZO's indlist and objlist
            self.__BZO[self.CalledIndex]=NewMat


        # Define standard operarations for zero object. In all these aspects, ZeroObj acts as a matrix of shape self.__shape() with all zeros. 
        def __str__(self):
            return str(self.__Mat)

        def __repr__(self):
            return self.__str__()


        def __add__(self,x):
            return self.__Mat + x

        def __radd__(self,x):
            return self+x

        def __mul__(self,x):
            return self+0

        def __rmul__(self,x):
            return self * x


        def __sub__(self,x):
            return self+(-x)

        def __rsub__(self,x):
            return (-1)*self + x

        def truediv(self,x):
            return (1/x)*self



    def __setitem__(self,Index,Value):
        """ 
        Set method. If Index is in IndList, sets self.__ObjList[z] = Value,
        where IndList[z]=Index Otherwise adds Index to IndList first (and a
        corresponding empty slot in ObjList), and then sets sets
        self.__ObjList[z] = Value, where IndList[z]=Index       
        """
        
        # Check that the value has the right shape and format
        
        Value=array(Value)
        if (not type(Value)==ndarray) or (not shape(Value)==self.__shape):
            raise ValueError("Assigned value must be ndarray of shape %s"%str(self.__shape))

        # Determine where ObjList should be updated. 
        Q = 3*max([abs(n) for n in Index])

        # Update IndexCounter and NumList if maximal integer in Index exceeds IndexCounter/3
        if Q>self.__IndexCounter:
            self.__Set_NumList(IndexCounter=Q)

        # Find the index z where IndList[z]=Index
        n = self.__FindIndex(Index)

        # If search was succesful, update corresponding slot in ObjList
        if n >=0:
            self.__ObjList[n]=Value

        #Otherwise, insert Index, Value and IndToNum(Index) in the appropriate slot in IndList, ObjList and NumList
        else:
            
            # Slot where new elements should go
            newIndex = -n-1

            self.__IndList.insert(newIndex,Index)
            self.__ObjList.insert(newIndex,Value)
            self.__NumList.insert(newIndex,self.__IndToNum(Index))

    def __SortLists(self):
        """
        Sorts indlist, objlist and numlist according to their NumList values.
        """
        
        AS=argsort(self.__NumList)

        self.__IndList=[self.__IndList[i] for i in AS]#list(self.__IndList[AS])
        self.__ObjList=[self.__ObjList[i] for i in AS]#list(self.__IndList[AS])
        self.__NumList=[self.__NumList[i] for i in AS]

    def NNZ(self):
        """
        Returns number of nonzero elements in ObjList
        """
        return len(self.__IndList)

    def ObjList(self):
        """ Returns ObjList"""
        return self.__ObjList[0:]*1

    def IndList(self):
        """ Returns IndList"""

        return self.__IndList[0:]*1  #Uses 0: to ensure that returned object is plain data, not a reference to self.__IndList

    def __mul__(self,y):
        """ 
        Multiplication. If F1(k) is multiplied with scalar lambda, returns lambda * F(k)
        If mulitplied with BZO of same shape, returns Out(k) = F1(k)*F2(k)"""

        # Determine whether BZO multiplication or scalar multiplication is used 
        
        if type(y)==type(self):
            # BZO multiplication:
            
            # Create output BZO
            Out = self._CreateSameType()

            # Set ObjList and IndList of output BZO by convolving lists of the two multiplies BZOs 
            for Ind1 in self.IndList():
                
                Obj1=self[Ind1]
                
                for Ind2 in y.IndList():
                    Obj2=y[Ind2]

                    Ind3 = tuple(add(Ind1,Ind2))
                    Out[Ind3] += Obj1*Obj2


        else:
            # Scalar multiplication
            
            # Create output BZO
            Out = self._CreateSameType()

            # Set output ObjList by multiplying elements in input ObjList with scalar
            Out.SetLists(self.IndList(),[y*x for x in self.__ObjList])

        return Out


    def __rmul__(self,x):
        return self*x


    def __truediv__(self,y):
        """Division: only division with scalars is defined""" 
        return self*(1/y)


    def __add__(self,Obj):
        """Addition of BZO with another BZO X""" 
        
        # Ensure that BZO and X are of the same type 
        if not type(self)==type(Obj):
            raise ValueError("Two added objects must be of the same type. Type of arguments are "+str(self.__class__.__name__)+", "+str(Obj.__class__.__name__))

        # Create output BZO of the same type
        Out = self._CreateSameType()

        # Initially, output BZO is identical to BZO. 
        ObjListOut = self.ObjList()
        IndListOut = self.IndList()

        # Get IndList and ObjList from X
        IndList2 = Obj.IndList()
        ObjList2 = Obj.ObjList()

        # Determine whether NumList of BZO and X are set with the same IndexCounter
        if self.__IndexCounter == Obj.__IndexCounter:
            # If the index counters are the same, we use a fast method, where we iterate the over the NumLists for BZO and X simultaneously
            
            # Shorthand for NumLists of BZO and X
            N1=1*self.__NumList
            N2=1*Obj.__NumList
            
            NNZ = len(N1)
            NNZ2 = len(N2)

            # Iteration counters for the two lists 
            z1 = 0
            z2=0
            
            # Iterate over elements in N1 and N2
            while z2 < NNZ2 and z1 < NNZ:
                Num2 = N2[z2]

                if Num2 ==N1[z1]:                    
                    # If N2[z2]=N1[z1], the index IndList1[z1] appers in both IndLists. Then ObjListOut[z1] should be given by sum of two corresponding objects. 

                    ObjListOut[1*z1]=1*(ObjListOut[z1]+ObjList2[z2])
                    z2+=1   # Look for next nonzero index in X. 

                elif Num2 < N1[z1]:
                    # If N2[z2]< N1[z1], N2[z2] is between N1[z1-1] and N1[z1]. Then ObjList2[z2] and IndList2[z2] should be inserted between z1-1 and z1 in ObjListOut

                    ObjListOut.insert(z1,ObjList2[z2])
                    IndListOut.insert(z1,IndList2[z2])
                    N1.insert(z1,Num2)

                    z2+=1   # Look for the next nonzero elements in X
                    NNZ+=1  # Number of nonzero elements in Output is increased by 1
                    z1+=1   # Whatever number comes next in N2, N2[z2+1]>ObjListOut[z1]=N2[z2]. We might as well increase z1 here then. 
                    

                elif Num2 > N1[z1]:
                    # If N2[z2] is larger than OutList[z1], we should incrase z1 until N2[z2]<=OutList[z1]
                    z1+=1

            # If there are still elements in N2List after z2 after the iteration, simply add them to the output. 
            ObjListOut = ObjListOut + ObjList2[z2:]
            IndListOut = IndListOut + IndList2[z2:]

            # Update Output BZO with new lists, and clean up. 
            Out.SetLists(IndListOut,ObjListOut)
            Out.CleanUp()
            return Out


        else:
            # If the index counters are not the same, add the two BZO's using the elementwise __set__ method
            Out.SetLists(IndListOut,ObjListOut)

            for Ind in IndList2:

                Out[Ind]+=Obj[Ind]

            Out.CleanUp()
            
            return Out

    def __radd__(self,Obj):

        return self + Obj

    def _CreateSameType(self):
        """ Creates empty BZO of same type as self"""
        return BZO(shape=self.__shape,dtype=self.__dtype)

    def __sub__(self,Obj):

        return (-1)*Obj + self

    def __rsub__(self,Obj):

        return self-Obj

    def __InputInterpreter(self,*args):
        """Interprets input from function call. Input can be of 3 types: VectorSpan, List, or Array"""

        if len(args)==self.__Dimension:
            VectorLists = [array(x,ndmin=1) for x in args]

            Format="VectorSpan"

            return [VectorLists,Format]

        elif len(args)==1:

            K = args[0]

            if type(K)==list:
                Karray=array(args[0]).T

                Format="List"
            else:
                Karray=K

                Format="Array"


            if (not type(Karray)==ndarray) or (not shape(Karray)[0]==self.__Dimension):
                raise IndexError("Argument must be ndarray of dimension %d x *"%self.__Dimension )

            return [Karray,Format]
        else:
            raise ValueError("Did not understand input")


    def __FindIndexSpan(self):

        self.__IndexSpanList=[]
        self.__IndexSpanPointerList=[]

        for d in range(0,self.__Dimension):

            List = list(set([Ind[d] for Ind in self.__IndList]))
            List.sort()

            self.__IndexSpanList.append(List)

            IndexSpanPointer  = [searchsorted(List,Ind[d]) for Ind in self.__IndList]

            self.__IndexSpanPointerList.append(IndexSpanPointer)



    def __call__(self,*args):

        """ Evaluate function at kvec (kvec is a momentum or array of momenta).
        Returns value, or array of values"""

        [Input,Format]=self.__InputInterpreter(*args)


        if Format=="List" or Format=="Array":
            Karray =Input


            OutShape = shape(Karray)[1:]

            Karray_dimension=len(OutShape)

            OutMatShape = self.__shape+OutShape
            OutMat = zeros(OutMatShape,dtype=self.__dtype)




            ## List argument to be passed to Einstein summation
            EinsteinList = [x for x in range(0,Karray_dimension+1)]

            ## Generate lists with exponential vectors


            for n in range(0,self.NNZ()):

                Ind = array(self._BZO__IndList[n])
                C   = self._BZO__ObjList[n]


                PhaseMat = exp(-1j*einsum(Ind,[0],Karray,EinsteinList))

                if not self.__shape ==():

                    dO = multiply.outer(C,PhaseMat)
                else :
                    dO = C*PhaseMat

                OutMat = OutMat + dO




            if Format=="Array":
                return OutMat

            elif Format=="List":
                return list(OutMat)

        elif Format=="VectorSpan":

            Vectors= Input

            OutShape = tuple([len(v) for v in Vectors])


            OutDimension=len(OutShape)

            OutMatShape = self.__shape+OutShape
            OutMat = zeros(OutMatShape,dtype=self.__dtype)


            self.__FindIndexSpan()

            VectorList=[]
            MultList=[]

            Module.OutArrayList= []
            for d in range(0,self.__Dimension):
                IS = self.__IndexSpanList[d]
                V = Vectors[d]
                MultList.append( [exp(-1j*V*Index) for Index in IS])


            Module.OutArrayList = [zeros(OutShape[-d:],dtype=complex) for d in range(2,self.__Dimension+1)]
            dO = zeros(OutMatShape,dtype=complex)


            def Multiply(Args):
                Nargs = len(Args)
                if Nargs>1:

                    return multiply.outer(Args[0],Multiply(Args[1:]),out=Module.OutArrayList[Nargs-2])
                else:
                    return Args[0]
#


            for n in range(0,self.NNZ()):

                C   = self._BZO__ObjList[n]


                VecList=[MultList[d][self.__IndexSpanPointerList[d][n]] for d in range(0,self.__Dimension)]


                PhaseMat = Multiply(VecList)

                if not self.__shape ==():

                    multiply.outer(C,PhaseMat,out = dO)

                else :
                    multiply(C,PhaseMat,out = dO)

                OutMat = OutMat + dO

#                print(PhaseMat)

            return OutMat


    def Gradient(self):
        """ Returns gradient \partial _{k_i} F(k) .
        BZO.Gradient() Returns BZO field of dimension (Dimension,)+shape(BZO), with
        BZO.Gradient()(k)[i,:,..] =\partial _{k_i} F(k)"""

        OutList  = []
        for dim in range(0,Dimension):
            X = self._CreateSameType()

            for Ind in self.IndList():

                Factor = -1j * Ind[dim]

                X[Ind] = Factor * self[Ind]

            OutList.append(X)

        return BZO(array(OutList))


    def shape(self):
        """ Returns shape of BZO"""
        return self.__shape


    def dtype(self):
        """Returns data type of BZO"""
        return self.__dtype

    def __repr__(self):

        if self.NNZ()>0 and self.NNZ ()< 20:

            Str = "Type: %s"%str(self.__class__.__name__)+"\n\n"

            for Ind in self.__IndList:
                Str += str(Ind)+": \n \n"
                Str += self[Ind].__str__()
                Str += "\n \n \n"


        elif self.NNZ()==0:
            Str = "%s object with all zeros"%str(self.__class__.__name__)

        elif self.NNZ()>20:
            Str = "%s of shape %s with %d Matrices (too long to show here) " %(self.__class__.__name__,str(self.shape()),self.NNZ())

        return Str

    def __str__(self):
        return self.__repr__()


    def conj(self):
        """ Returns conjugate transpose of BZO"""

        Out = self._CreateSameType()

        for Ind in self.IndList():
            OutInd = tuple(-x for x in Ind)

            Out[OutInd]=self[Ind].conj().T

        return Out

    def slice(self,*Indices):
        """ Return BZO such that, with Y=BZO.slice(Indices),  Y(k) = BZO(k)(Indices)."""

        Ind = tuple(Indices)


        try:

            OutShape=shape((1*self[(0,)*Dimension])[Indices])
        except:
            raise IndexError("Wrong format for indices")

        Out = BZO(shape=OutShape)

        for Ind in self.IndList():

            Out[Ind]=array(self[Ind][Indices])

        Out.CleanUp()

        return Out

    def __delitem__(self,Ind):

        n=self.__FindIndex(Ind)

        if not n==None :

            del self.__IndList[n]
            del self.__ObjList[n]
            del self.__NumList[n]

    def CleanUp(self):
        """ Remove negligible elements in IndList and ObjList. By default,
        elements are discared, if they have max-norm less than 1e-10
        """
        for Ind in self.IndList():
            if amax(abs(self[Ind]))<1e-10:
                del self[Ind]



    def Norm(self):
        """ Returns norm \int d^D k \Tr (F^\dagger(k) F(k))"""

        return sqrt(sum([sum(abs(x)**2) for x in self.__ObjList]))

def Karray(*args):
    """TBA      """
    return None

def MergeBZOs(FieldArray):
    """Combine field from array of fields"""
    shape0=shape(FieldArray)

    FlatArray = FieldArray.flatten()
    N_elements = len(FlatArray)

    try:

        shape1=FlatArray[0].shape()
        type1=type(FlatArray[0])


        SameType = prod([type(x)==type1 for x in FlatArray])
        SameShape = prod([x.shape()==shape1 for x in FlatArray])


        if not (SameType and SameShape):
            raise ValueError("Argument must be array of objects of the same type and shape")
    except:
        raise ValueError("Argument must be array of objects of the same type and shape")


    OutShape = tuple(shape1)+tuple(shape0)

    Out = BZO(shape=OutShape)




    ### Construct IndList


    SliceTuple=tuple(slice(x) for x in shape1)

    for n in range(0,N_elements):
        field = FlatArray[n]

        IndList=field.IndList()


        for Ind in IndList:

            MatrixIndex = unravel_index(n,shape0)

            Out[Ind][SliceTuple+MatrixIndex]+=field[Ind]

    return Out


class Hamiltonian(BZO):

    """Physical Hamiltonian object. Constains information about the translationally invariant part of the Hamiltonian"""

    def __init__(self,*args):


        ArgV=[arg for arg in args]

        if len(ArgV)==1:

            Obj = ArgV[0]


            if (not type(Obj)==Module.BZO) or (not Obj.shape() ==(OrbitalDimension,OrbitalDimension)):
                raise TypeError("Argument must be a field of dimension %d x %d "%(OrbitalDimension,OrbitalDimension))

            BZO.__init__(self,shape=(OrbitalDimension,OrbitalDimension))
            self._BZO__Dimension = Dimension
            self.SetLists(Obj.IndList(),Obj.ObjList())
#            self._BZO__Zero = BZO.__ZeroObject(self)


            if not self.IsHermitian():
                print("Warning: Hamiltonian is non-hermitian")

        elif len(ArgV)==0:
            if not (type(shape)==tuple or type(shape)==int):
                raise ValueError("Shape must be given by an integer or a tuple")

            self.__Dimension=Dimension

            self.SetLists([],[])

            self.__shape=shape
            self.__dtype=dtype

            self.__Zero = self.__ZeroObject(self)

        else:
            raise ValueError("BZO can only take a single array of other BZO's as argument")


#        BZO.__init__(self,shape=(OrbitalDimension,OrbitalDimension))


    def Bands(self,Karray):
        """ Compute energy band structure at k-points specified by Karray"""

        B = self(Karray)

        OutShape = (OrbitalDimension,)+shape(Karray)[1:]
        Nk = int(prod(shape(Karray)[1:])+0.1)

        BFlatShape=(OrbitalDimension,OrbitalDimension,Nk)

        Bflat = reshape(B,BFlatShape)

        Bands = array([eigh(Bflat[:,:,n].T,eigvals_only=True) for n in range(0,Nk)]).T
        Bands = reshape(Bands,OutShape)

        return Bands


    def Trace(self):
        """Compute trace of self. Returns Scalar object"""

        Out = Scalar()

        for Ind in self.IndList():
            Out[Ind]=trace(self[Ind])


        return Out

    def _CreateSameType(self):
        return Hamiltonian()

    def IsHermitian(self):
        """Determine whether Hamiltonian is Hermitian"""

        Hermitian=True
        for Ind in self.IndList():
            Q=tuple(-x for x in Ind)

            X = self[Ind].conj().T-self[Q]

            A=amax(list(abs(X.flatten())))


            if A > 1e-9:
                Hermitian=False

        return Hermitian



class Scalar(BZO):
    """ Sxcalar-valued function on the BZ"""

    def __init__(self,*args):
        if len(args)==2:
            BZO.__init__(self,*args)
        elif len(args)==0:
            BZO.__init__(self,shape=())


    def __getitem__(self,Index):
        return BZO.__getitem__(self,Index)*1

    def __setitem__(self,Index,Value):
        BZO.__setitem__(self,Index,array(Value))


    def __add__(self,y):

        if not (type(y)==Module.Scalar or type(y)==Module.BZO or type(y)==Module.Hamiltonian):

            Out = Scalar(self.IndList(),self.ObjList())
#
#            Out._BZO__ObjList=self.ObjList()
#            Out._BZO__IndList=self.IndList()
#

            Out[(0,)*Dimension]=Out[(0,)*Dimension]*1+y

        else:
            Out = BZO.__add__(self,y)

        return Out


    def __radd__(self,y):
        return self+y

    def __sub__(self,y):

        return self + (-1*y)

    def __rsub__(self,y):
        return self-y

    def __mul__(self,y):

        if type(y) == ndarray:

            Out = BZO(shape=shape(y))

            for Ind in self._BZO__IndList:

                Out[Ind] = self[Ind]*y


            return Out

        else:

            return BZO.__mul__(self,y)




    def __rmul__(self,y):
        return self*y

    def _CreateSameType(self):
        return Scalar()


def DataToScalar(Data,Cutoff=1e-3):
    """ Turn data into scalar object. Discard scalar elements smaller than cutoff"""

    import numpy.fft as fft

    # Renormalizing cutoff
    Cutoff = Cutoff * sqrt(sum(abs(Data)**2)/prod(shape(Data)))

    DataDimension = shape(Data)
    FourierData=fft.fftn(Data)/prod(DataDimension)

    RawIndList=where(abs(FourierData)>Cutoff)

    Nind = len(RawIndList[0])

    def FindIndex(Index):
        Out =[]
        for n in range(0,len(Index)):# in Index:
            x = Index[n]
            Dim=DataDimension[n]
            y = int((x+Dim/2)%Dim - Dim/2 )

            Out.append(-y)

        return tuple(Out)

    IndList= []
    ObjList = []

    print("Converting data to Scalar object. Number of nonzero elements: %d"%Nind)
    for n in range(0,Nind):

        Raw_index = tuple(int(A[n]+0.1) for A in RawIndList)

        Ind = FindIndex(Raw_index)
        IndList.append(Ind)

        ObjList.append(FourierData[Raw_index])



    Out=Scalar(IndList,ObjList)
    print("")
    return Out

def GetTranslationOperators():
    OutList=[]
    for d in range(0,Dimension):
        X = Scalar()

        Dir = tuple(d*[0]+[1]+[0]*(Dimension-d-1))

        X[Dir]=1

        OutList.append(X)

    return OutList



