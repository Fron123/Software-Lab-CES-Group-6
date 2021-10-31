# Software Lab CES

# Table of Contents
1. [Constant Aware Sparse Newton Solver](#solver)
2. [Installation Guide](#installation-guide) <br>
            2.1 [Download solver](#download-solver) <br>
            2.2 [Necessary third party software](#necessary-third-party-software) <br>
            2.3 [Build the solver](#build-the-solver)
3. [Usages](#usage)
&nbsp;


# Solver

The software computes the solution of a differentiable parametrized sparse system of nonlinear equations with the residual F(x,p) = 0 using Newton’s method F'(x_{k})*\Delta(x_{k})=-F(x_{k}) with: Delta(x_{k}) = x_{k+1}-x_{k} and a convex unconstrained minimization problem min(f(x,p)) , f'(x,p) = 0 by exploiting the sparsity of the Jacobian and its variable and constant submatrices. This finds usage in finding for example a solution for KKT systems see e.g


Installation Guide
======================================
This Software is based on C++, so you first need a compatible system. 

Download solver
--------------------------------------
If you already got this, you can download this repository directly over this site. If you want to istall this Software directly over the Linux terminal you must follow these instruction:

    git clone https://github.com/Fron123/Software-Lab-CES-Group-6.git  
   

Necessary third party software
--------------------------------------------
This solver uses different third party softwares which need to be installed. First of all we use the Software dco for the computation of the derivatives. This software can be downloaded on the Website:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.nag.com/content/downloads-dco-c-versions  
 
For the compression we use the software ColPack. This Software allows us to minimize our sparse matrices with graph coloring. you will find a further explination at the repository of ColPack. There you can also download the software. You find that repository under:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/CSCsw/ColPack  
    
If you want to use the Terminal follow these instructions:
     
    cd              
    git clone https://github.com/CSCsw/ColPack.git  
    cd ColPack                                       
    
And finally you will need the software Eigen. Eigen is used for the computation of the Linear systems. In our case we use the Spase LU solver of Eigen. You can download Eigen on the Website:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://eigen.tuxfamily.org/index.php?title=Main_Page

Build the solver
---------------------------------------------

Before you can build the solver you must edit the Makefile. It is necessary that the path are right, if there aren't correct the solver could not be build. You must edit the main Makefile. Here you have a little Example: 

            COLPACK_ROOT = $(HOME)/Software/ColPack
            EIGEN_DIR=$(HOME)/Software/Eigen
            DCO_DIR=$(HOME)/Software/dco
            BASE_DIR=$(HOME)/Dokumente/SP_CES/Code    
          
In this Example you can see the third party software are all together in one Folder, that is recommended. If you have installed e.g. ColPack in the Path /Documents/Stuff/Colpack,
then you must change the path to $(HOME)/Documents/Stuff/ColPack. The same goes for the others paths. 

Usages
---------------------------------------------
The software will be able to solve optimaization problems, i.e to compute minimum or maximum or stationary point/s of a function using numerical methods.
A real world use of our software can be in operation of a power plant to maximize the costs. Where the physical entites such as mass or density of the material can be our input vector x and other physical contraints such as thermodynamic laws, conversation laws can be our parameter vector.
