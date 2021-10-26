# Software Lab CES

# Table of Contents
1. [Constant Aware Sparse Newton Solver](#solver)
2. [Installation Guide](#installation-guide)
            2.1 [Download solver](#download-solver)
            2.2 [Necessary third party software](#necessary-third-party-software)
            2.3 [Build the solver](#build)
3. [Usages](#usage)
&nbsp;


# Solver

Some Discription of the solver should look like the one in the Report


Installation Guide
======================================
This Software is based on C++, so you first need a compatible system. 

Download solver
--------------------------------------
If you already got this, you can download this repository directly over this site. If you want to istall this Software directly over the Linux terminal you must follow these instruction:

    git clone https://github.com/Fron123/Software-Lab-CES-Group-5.git   #Download Software
   

Necessary third party software
--------------------------------------------
This solver uses different third party softwares which need to be installed. First of all we use the Software dco for the computation of the derivatives. This software can be downloaded on the Website:

    https://www.nag.com/content/downloads-dco-c-versions   #Download dco
 
For the compression we use the software ColPack. This Software allows us to minimize our sparse matrices with graph coloring. you will find a further explination at the repository of ColPack. There you can also download the software. You find that repository under:

    https://github.com/CSCsw/ColPack   #Repository ColPack
    
If you want to use the Terminal follow these instructions:
     
    cd              
    git clone https://github.com/CSCsw/ColPack.git   #Download ColPack
    cd ColPack                   # go to ColPack Root Directory
