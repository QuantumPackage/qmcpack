#!/usr/bin/env python3

print ("#QP -> QMCPACK")

# ___           
#  |  ._  o _|_ 
# _|_ | | |  |_ 
#

from ezfio import ezfio

import os
import sys
import functools
import numpy as np
ezfio_path = sys.argv[1]

ezfio.set_file(ezfio_path)

try:
   PBC=ezfio.get_qmcpack_qmc_pbc()
   Complex=True 
except IOError:
   PBC=False
   Complex=False

do_pseudo_pbc=False
do_pseudo=False

print("PBC=",PBC)
if PBC:
   do_pseudo_pbc = ezfio.get_qmcpack_qmc_pseudo()
else:
   do_pseudo = ezfio.get_pseudo_do_pseudo()
   if do_pseudo:
#       print ("do_pseudo True")
       from qp_path import QP_ROOT

       l_ele_path = os.path.join(QP_ROOT,"data","list_element.txt")
       with open(l_ele_path, "r") as f:
          data_raw = f.read()

       l_element_raw = data_raw.split("\n")
       l_element = [element_raw.split() for element_raw in l_element_raw]
       d_z = dict((abr, z) for (z, abr, ele, _) in filter(lambda x: x != [], l_element) )
   print ("do_pseudo=",do_pseudo)

try:
    n_det = ezfio.get_determinants_n_det()
except IOError:
    n_det = 1
Multidet=False
if n_det == 1:
    print ("multi_det False")
else:
    print ("multi_det True")
    Multidet=True

Cartesian=True
if PBC:
    Cartesian=ezfio.get_qmcpack_qmc_cart()

#             
# |\/| o  _  _ 
# |  | | _> (_ 
#

def list_to_string(l):
    return " ".join(map(str, l))


ao_num = ezfio.get_ao_basis_ao_num()
print ("ao_num", ao_num)

mo_num = ezfio.get_mo_basis_mo_num()
print ("mo_num", mo_num)

alpha = ezfio.get_electrons_elec_alpha_num()
beta = ezfio.get_electrons_elec_beta_num()
spin =  2 * (alpha - beta) + 1
print ("elec_alpha_num", alpha)
print ("elec_beta_num", beta)
print ("elec_tot_num", alpha + beta)
print ("spin_multiplicity", spin )

l_label = ezfio.get_nuclei_nucl_label()
l_charge = ezfio.get_nuclei_nucl_charge()
l_coord = ezfio.get_nuclei_nucl_coord()
if PBC:
  Nbkpts=ezfio.get_nuclei_kpt_num()

l_coord_str = [list_to_string(i) for i in zip(*l_coord)]
natom=len(l_label)
print ("Number of Atoms ", natom )

#  _               
# /   _   _  ._ _| 
# \_ (_) (_) | (_| 
#
print ("Atomic coord in Bohr")
for i, t in enumerate(zip(l_label, l_charge, l_coord_str)):
    t_1 = d_z[t[0]] if do_pseudo else t[1]
    t_new = [t[0],t_1,t[2]]
    print (list_to_string(t_new))



#  _                 __        
# |_)  _.  _ o  _   (_   _ _|_ 
# |_) (_| _> | _>   __) (/_ |_ 
#
import re
if not PBC :
#
# Call externet process to get the sysmetry
#
  import subprocess
  process = subprocess.Popen(
      ['qp_print_basis', ezfio_path],
      stdout=subprocess.PIPE)
  out, err = process.communicate()

  basis_raw, sym_raw, _ = out.decode().split("\n\n\n")
  basis_without_header = "\n".join(basis_raw.split("\n")[19:])
  l_basis_raw = re.split('\n\s*\n', basis_without_header)
  a_already_print = []
  l_basis_clean = []
  for i, (a,b) in enumerate(zip(l_label,l_basis_raw)):
      if a not in a_already_print:
          l_basis_clean.append(b.replace('Atom {0}'.format(i + 1), a))
          a_already_print.append(a)
      else:
          continue
  #print ("BEGIN_BASIS_SET\n")
  #print ("\n\n".join(l_basis_clean))
  #print ("END_BASIS_SET")

#       _     
# |\/| / \  _ 
# |  | \_/ _> 
#


#
# Function
#
d_gms_order ={ 0:["s"],
     1:[ "x", "y", "z" ],
     2:[ "xx", "yy", "zz", "xy", "xz", "yz" ],
     3:[ "xxx", "yyy", "zzz", "xxy", "xxz", "yyx", "yyz", "zzx", "zzy", "xyz"],
     4: ["xxxx", "yyyy", "zzzz", "xxxy", "xxxz", "yyyx", "yyyz", "zzzx", "zzzy", "xxyy", "xxzz", "yyzz", "xxyz", "yyxz", "zzxy", "xxxx", "yyyy", "zzzz", "xxxy", "xxxz", "yyyx", "yyyz", "zzzx", "zzzy", "xxyy", "xxzz", "yyzz", "xxyz", "yyxz","zzxy"], 
     5:["xxxxx","yyyyy","zzzzz","xxxxy","xxxxz","yyyyx","yyyyz","zzzzx","zzzzy","xxxyy","xxxzz","yyyxx","yyyzz","zzzxx","zzzyy","xxxyz","yyyxz","zzzxy","xxyyz","xxzzy","yyzzx"],
     6:["xxxxxx","yyyyyy","zzzzzz","xxxxxy","xxxxxz","yyyyyx","yyyyyz","zzzzzx","zzzzzy","xxxxyy","xxxxzz","yyyyxx","yyyyzz","zzzzxx","zzzzyy","xxxxyz","yyyyxz","zzzzxy","xxxyyy","xxxzzz","yyyzzz","xxxyyz","xxxzzy","yyyxxz","yyyzzx","zzzxxy","zzzyyx","xxyyzz"],
}

def mo_k2gamma(phase,mo_energy):                                              
    
    c0 = np.array(ezfio.get_mo_basis_mo_coef_kpts()).view(dtype=np.complex128)
    mo_coeff = c0.reshape(c0.shape[:-1]).transpose((0,2,1))
    E_g = np.hstack(mo_energy)                                                                                        
    C_k = np.asarray(mo_coeff)                                                                                        
   
    Nk, Nao, Nmo = C_k.shape
    NR = phase.shape[0]
    # Transform AO indices
    C_gamma = np.einsum('Rk, kum -> Rukm', phase, C_k)                                                                
    C_gamma = C_gamma.reshape(Nao*NR, Nk*Nmo)
    
    E_sort_idx = np.argsort(E_g)
    E_desort_idx=np.argsort(  E_sort_idx ) 
    E_g = E_g[E_sort_idx]
    C_gamma = C_gamma[:,E_sort_idx]
    

    C_gamma_unsorted=C_gamma[:,E_desort_idx]                                                                         
    E_g_unsorted=E_g[E_desort_idx]
    return E_g, C_gamma, E_g_unsorted,C_gamma_unsorted                                                                


def compare_gamess_style(item1, item2):
    def cmp(a, b):
        return (a > b) - (a < b) 
    item1=item1[2]
    item2=item2[2]
    n1,n2 = map(len,(item1,item2))
    assert (n1 == n2)
    try:
        l = d_gms_order[n1]
    except KeyError:
        return 0
#       raise (KeyError, "We dont handle L than 4")
    else:
        a = l.index(item1)
        b = l.index(item2)
        return cmp( a, b )

def expend_sym_str(str_):
    #Expend x2 -> xx
    # yx2 -> xxy
    for i, c in enumerate(str_):
        try:
            n = int(c)
        except ValueError:
            pass
        else:
            str_ = str_[:i - 1] + str_[i - 1] * n + str_[i + 1:]

    #Order by frequency
    return "".join(sorted(str_, key=str_.count, reverse=True))


def expend_sym_l(l_l_sym):
    for l in l_l_sym:
        l[2] = expend_sym_str(l[2])

    return l_l_sym


def n_orbital(n):
    if n==0:
        return 1
    elif n==1:
        return 3
    else:
        return 2*n_orbital(n-1)-n_orbital(n-2)+1

def get_nb_permutation(str_):
    if (str_) == 's': return 1
    else: return n_orbital(len(str_))

def order_l_l_sym(l_l_sym):
    n = 1
    iter_ = range(len(l_l_sym))
    for i in iter_:
        if n != 1:
            n += -1
            continue 

        l = l_l_sym[i]
        n = get_nb_permutation(l[2])

        l_l_sym[i:i + n] = sorted(l_l_sym[i:i + n],
                                  key=functools.cmp_to_key(compare_gamess_style))


    return l_l_sym


#==========================
# We will order the symetry
#==========================
if not PBC :
 l_sym_without_header = sym_raw.split("\n")[3:-2]
 l_l_sym_raw = [i.split() for i in l_sym_without_header]
 l_l_sym_expend_sym = expend_sym_l(l_l_sym_raw)
 l_l_sym_ordered = order_l_l_sym(l_l_sym_expend_sym)

#========
#MO COEF
#========
def order_phase(mo_coef):
    #Order
    mo_coef_phase = []
    import math

    for i in mo_coef:
        if abs(max(i)) > abs(min(i)):
            sign_max = math.copysign(1, max(i))
        else:
            sign_max = math.copysign(1, min(i))

        if sign_max == -1:
            ii = [-1 * l for l in i]
        else:
            ii = i

        mo_coef_phase.append(ii)
    return mo_coef_phase

def chunked(l, chunks_size):
    l_block = []
    for i in l:
        chunks = [i[x:x + chunks_size] for x in range(0, len(i), chunks_size)]
        l_block.append(chunks)
    return l_block



def mo_coef_H5(mo_coef, l_l_sym):
    
    a = mo_coef
    orderd_mo_coeff=[]
    print ("size ",len(l_l_sym))
    for m in range(len(mo_coef)):
        counter=0
        new=[]
        for l in l_l_sym:
            i = int(l[0]) - 1
            i_a = int(l[1]) - 1
            sym = l[2]

            new.append(a[m][i])
            counter+=1
        orderd_mo_coeff.append(new) 
    return orderd_mo_coeff 

if PBC :
  mo_coef_all     = np.array(ezfio.get_mo_basis_mo_coef_complex()).transpose((2,1,0))
  mo_coef         = mo_coef_all[0]
  mo_coef_complex = mo_coef_all[1]

  #Reordered by kpts
  mo_coef_kpts_all      = np.array(ezfio.get_qmcpack_mo_coef_complex_reordered()).transpose((2,1,0))
  mo_coef_kpts          = mo_coef_kpts_all[0]
  mo_coef_kpts_complex  = mo_coef_kpts_all[1]

else :
  mo_coef = ezfio.get_mo_basis_mo_coef()
  orderd_mo_coeff=[]
  orderd_mo_coeff=mo_coef_H5(mo_coef, l_l_sym_ordered)

#  _                    
# |_) _  _       _|  _  
# |  _> (/_ |_| (_| (_) 
#
if do_pseudo:
#    print ("")
#    print ("BEGIN_PSEUDO")
    klocmax = ezfio.get_pseudo_pseudo_klocmax()
    kmax = ezfio.get_pseudo_pseudo_kmax()
    lmax = ezfio.get_pseudo_pseudo_lmax()

    n_k = ezfio.get_pseudo_pseudo_n_k()
    v_k = ezfio.get_pseudo_pseudo_v_k()
    dz_k = ezfio.get_pseudo_pseudo_dz_k()

    n_kl = ezfio.get_pseudo_pseudo_n_kl()
    v_kl = ezfio.get_pseudo_pseudo_v_kl()
    dz_kl = ezfio.get_pseudo_pseudo_dz_kl()

    for i, a in enumerate(l_label):

        l_str = []

        #Local
        l_dump = []
        for k in range(klocmax):
            if v_k[k][i]:
                l_ = list_to_string([v_k[k][i], n_k[k][i] + 2, dz_k[k][i]])
                l_dump.append(l_)

        l_str.append(l_dump)

        #Non local
        for l in range(lmax + 1):
            l_dump = []
            for k in range(kmax):
                if v_kl[l][k][i]:
                    l_ = list_to_string([v_kl[l][k][i], n_kl[l][k][i] + 2,
                                         dz_kl[l][k][i]])
                    l_dump.append(l_)
            if l_dump:
                l_str.append(l_dump)



#  _         
# | \  _ _|_ 
# |_/ (/_ |_ 
#
nexcitedstate = ezfio.get_determinants_n_states()
print("nexcitedstate=",nexcitedstate)
if PBC:
#  psi_coef_all =ezfio.get_determinants_psi_coef_complex()
#  psi_coef         = [[0 for x in range(n_det)] for y in range(nexcitedstate)] 
#  psi_coef_complex = [[0 for x in range(n_det)] for y in range(nexcitedstate)] 
#
#  for i in range(nexcitedstate):
#     for j in range(n_det):
#        psi_coef[i][j]=psi_coef_all[i][j][0]
#        psi_coef_complex[i][j]=psi_coef_all[i][j][1]

  psi_coef_complex = np.array(ezfio.get_determinants_psi_coef_complex()).transpose((2,1,0))
  psi_coef         = psi_coef_complex[0]
  psi_coef_imag    = psi_coef_complex[1]


else :
  psi_coef = ezfio.get_determinants_psi_coef()

psi_det = ezfio.get_determinants_psi_det()
bit_kind = ezfio.get_determinants_bit_kind()

print ("det_num", n_det)
print ("")


encode = 8*bit_kind

def bindigits(n, bits):
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

decode = lambda det: ''.join(bindigits(i,encode)[::-1] for i in det)[:mo_num]

MultiDetAlpha = []
MultiDetBeta = []
for (det_a, det_b) in psi_det:                                                                
        MyDetA=decode(det_a)                                                                                          
        MyDetB=decode(det_b)                                                                                          
        #print (MyDetA)                                                                                               
        #print (MyDetB)                                                                                               
        #print ('')
        MultiDetAlpha.append(det_a)                                                                                   
        MultiDetBeta.append(det_b) 

d_l = {'S':0, 'P':1, 'D':2, 'F':3, 'G':4, 'H':5, 'I':6}

IonName=dict([('H',1),  ('He',2),  ('Li',3),('Be',4),  ('B', 5),  ('C', 6),  ('N', 7),('O', 8),  ('F', 9),   ('Ne',10),   ('Na',11),('Mg',12),   ('Al',13),   ('Si',14),   ('P', 15),   ('S', 16),('Cl',17),   ('Ar',18),   ('K', 19),   ('Ca',20),   ('Sc',21),   ('Ti',22),   ('V', 23),   ('Cr',24),   ('Mn',25),   ('Fe',26),   ('Co',27),   ('Ni',28),   ('Cu',29),   ('Zn',30),   ('Ga',31),   ('Ge',32),   ('As',33),   ('Se',34),   ('Br',35),   ('Kr',36),   ('Rb',37),   ('Sr',38),   ('Y', 39),  ('Zr',40),   ('Nb',41),   ('Mo',42),   ('Tc',43),   ('Ru',44),   ('Rh',45),   ('Pd',46),   ('Ag',47),   ('Cd',48),   ('In',49),   ('Sn',50),   ('Sb',51),   ('Te',52),   ('I', 53),   ('Xe',54),   ('Cs',55),   ('Ba',56),   ('La',57),   ('Ce',58), ('Pr',59),   ('Nd',60),   ('Pm',61),   ('Sm',62),   ('Eu',63),   ('Gd',64),   ('Tb',65),   ('Dy',66),   ('Ho',67),  ('Er',68),   ('Tm',69),   ('Yb',70),   ('Lu',71),   ('Hf',72),   ('Ta',73),   ('W', 74),   ('Re',75),   ('Os',76),   ('Ir',77),   ('Pt',78),   ('Au',79),   ('Hg',80), ('Tl',81),   ('Pb',82),  ('Bi',83),   ('Po',84),   ('At',85),   ('Rn',86),   ('Fr',87),   ('Ra',88),   ('Ac',89),   ('Th',90),   ('Pa',91),   ('U', 92),   ('Np',93)]) 


import h5py
from collections import defaultdict
title="QP2QMCACK"
H5_qmcpack=h5py.File(title+'.h5','w')

groupApp=H5_qmcpack.create_group("application")
CodeData  = groupApp.create_dataset("code",(1,),dtype="S15")
CodeData[0:] = b"Quantum Package"
CodeVer  = groupApp.create_dataset("version",(3,),dtype="i4")
CodeVer[0:] = 2
CodeVer[1:] = 0
CodeVer[2:] = 0


GroupPBC=H5_qmcpack.create_group("PBC")
GroupPBC.create_dataset("PBC",(1,),dtype="b1",data=PBC)


dt = h5py.special_dtype(vlen=bytes)
#Group Atoms
groupAtom=H5_qmcpack.create_group("atoms")

#Dataset Number Of Atoms
groupAtom.create_dataset("number_of_atoms",(1,),dtype="i4",data=natom)


#Dataset Number Of Species 
#Species contains (Atom_Name, Atom_Number,Atom_Charge,Atom_Core)

l_atoms = [ (l_label[x],IonName[l_label[x]],l_charge[x],l_charge[x]) for x in  range(natom)  ] 

#print l_atoms

d = defaultdict(list)
for i,t in enumerate(l_atoms):
      d[t].append(i)



idxSpeciestoAtoms = dict()
uniq_atoms= dict()
for i, (k,v) in enumerate(d.items()):
      idxSpeciestoAtoms[i] = v
      uniq_atoms[i] = k

idxAtomstoSpecies = dict()
for k, l_v in idxSpeciestoAtoms.items():
      for v in l_v:
          idxAtomstoSpecies[v] = k

NbSpecies=len(idxSpeciestoAtoms.keys())

groupAtom.create_dataset("number_of_species",(1,),dtype="i4",data=NbSpecies)

#Dataset positions 
MyPos=groupAtom.create_dataset("positions",(natom,3),dtype="f8")

Pos_Atm=[i for i in zip(*l_coord)] 
for i in range(natom):
  MyPos[i:]=Pos_Atm[i] 

#Group Atoms
for x in range(NbSpecies):
    atmname=str(uniq_atoms[x][0])
    groupSpecies=groupAtom.create_group("species_"+str(x))
    groupSpecies.create_dataset("atomic_number",(1,),dtype="i4",data=uniq_atoms[x][1])
    mylen="S"+str(len(atmname))
    strList=[atmname]
    asciiList = [n.encode("ascii", "ignore") for n in strList]
    groupSpecies.create_dataset('name', (1,),mylen, asciiList)
    groupSpecies.create_dataset("charge",(1,),dtype="f8",data=uniq_atoms[x][2])
    groupSpecies.create_dataset("core",(1,),dtype="f8",data=uniq_atoms[x][3])

SpeciesID=groupAtom.create_dataset("species_ids",(natom,),dtype="i4")

for x in range(natom):
      SpeciesID[x:]  = idxAtomstoSpecies[x]

#Parameter Group
GroupParameter=H5_qmcpack.create_group("parameters")
if PBC:
  GroupParameter.create_dataset("ECP",(1,),dtype="b1",data=bool(do_pseudo_pbc))
else:
  GroupParameter.create_dataset("ECP",(1,),dtype="b1",data=bool(do_pseudo))
bohrUnit=True

GroupParameter.create_dataset("Unit",(1,),dtype="b1",data=bohrUnit) 
GroupParameter.create_dataset("NbAlpha",(1,),dtype="i4",data=alpha) 
GroupParameter.create_dataset("NbBeta",(1,),dtype="i4",data=beta) 
GroupParameter.create_dataset("NbTotElec",(1,),dtype="i4",data=alpha+beta)
GroupParameter.create_dataset("spin",(1,),dtype="i4",data=spin-1) 
GroupParameter.create_dataset("Multidet",(1,),dtype="b1",data=Multidet) 
 

#basisset Group
GroupBasisSet=H5_qmcpack.create_group("basisset")
#Dataset Number Of Atoms
GroupBasisSet.create_dataset("NbElements",(1,),dtype="i4",data=NbSpecies)
strList=['LCAOBSet']
asciiList = [n.encode("ascii", "ignore") for n in strList]
GroupBasisSet.create_dataset('name', (1,),'S8', asciiList)

#atomicBasisSets Group
if PBC :

  qmc_totshell=ezfio.get_qmcpack_qmc_nshell() 
  qmc_nucl=ezfio.get_qmcpack_qmc_nucl() 
  qmc_lbas=ezfio.get_qmcpack_qmc_lbas() 
  qmc_expo=ezfio.get_qmcpack_qmc_expo() 
  qmc_prim_num_max=ezfio.get_qmcpack_qmc_prim_num_max()
  qmc_prim_num=ezfio.get_qmcpack_qmc_prim_num()
  qmc_coef=ezfio.get_qmcpack_qmc_coef()
  phase=ezfio.get_qmcpack_qmc_phase()
  qmc_phase = np.array(phase).view(dtype=np.complex128).reshape((Nbkpts,Nbkpts))


  qmc_nshell=[]


  n=0
  qmc_nshell        = [[0 for x in range(qmc_nucl.count(y))] for y in range(natom)]  
  for y in range(natom):
     for x in range(qmc_nucl.count(y)) :
       qmc_nshell[y][x]=n
       n+=1 

  for x in range(NbSpecies):
      MyIdx=idxSpeciestoAtoms[x][0]
      atomicBasisSetGroup=GroupBasisSet.create_group("atomicBasisSet"+str(x))
      mylen="S"+str(len(uniq_atoms[x][0]))
      
      strList=[uniq_atoms[x][0]]
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('elementType', (1,),mylen, asciiList)
      if Cartesian:
          strList=['cartesian']
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          atomicBasisSetGroup.create_dataset('angular', (1,),'S9', asciiList)

          strList=['Gamess']
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          atomicBasisSetGroup.create_dataset('expandYlm', (1,),'S6', asciiList)

      else:
          strList=['spherical']
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          atomicBasisSetGroup.create_dataset('angular', (1,),'S9', asciiList)

          strList=['pyscf']
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          atomicBasisSetGroup.create_dataset('expandYlm', (1,),'S5', asciiList)

      atomicBasisSetGroup.create_dataset("grid_npts",(1,),dtype="i4",data=1001)
      atomicBasisSetGroup.create_dataset("grid_rf",(1,),dtype="i4",data=100)
      atomicBasisSetGroup.create_dataset("grid_ri",(1,),dtype="f8",data=1e-06)



      strList=['log']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('grid_type', (1,),'S3', asciiList)
      strList=['gaussian']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('name', (1,),'S8', asciiList)
      strList=['no']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('normalized', (1,),'S2', asciiList)

      n=0
      nshell=qmc_nshell[MyIdx]
      for i in nshell:
         l = qmc_lbas[i]
         nprim=qmc_prim_num[i]
         contracted_coeffs = [[] for i in range(nprim)]  
         contracted_exp = [] 
         for mm in range(nprim):
             contracted_coeffs[mm].append(qmc_coef[mm][i])
             contracted_exp.append(qmc_expo[mm][i])
         for line in zip(*contracted_coeffs):
              BasisGroup=atomicBasisSetGroup.create_group("basisGroup"+str(n))
              mylen="S"+str(len((uniq_atoms[x][0]+str(n)+str(l))))
              strList=['Gaussian'] 
              asciiList = [n.encode("ascii", "ignore") for n in strList]
              BasisGroup.create_dataset('type',(1,),'S8',asciiList)
              strList=[uniq_atoms[x][0]+str(n)+str(l)] 
              asciiList = [n.encode("ascii", "ignore") for n in strList]
              BasisGroup.create_dataset('rid', (1,),mylen, asciiList)
              
            # # BasisGroup.create_dataset("Shell_coord",(3,),dtype="f8",data=loc_cell.bas_coord(i))
              BasisGroup.create_dataset("NbRadFunc",(1,),dtype="i4",data=nprim)
              Val_l=BasisGroup.create_dataset("l",(1,),dtype="i4",data=l)
              Val_n=BasisGroup.create_dataset("n",(1,),dtype="i4",data=n)
              RadGroup=BasisGroup.create_group("radfunctions")
              #print ("<basisGroup",n," rid=",uniq_atoms[x][0]+str(n)+str(l)," n=",n,"  l=",l ,"NbRadFunc=",nprim,"type=Gaussian>")
              IdRad=0
           
              for e,c in zip(contracted_exp,line):
                  DataRadGrp=RadGroup.create_group("DataRad"+str(IdRad))
                  DataRadGrp.create_dataset("exponent",(1,),dtype="f8",data=e)
                  DataRadGrp.create_dataset("contraction",(1,),dtype="f8",data=c)
                  #print  ("<radfunc exponent=",e," contraction=",c, "DataRad=",n,"IdRad=",IdRad,"/>")
                  IdRad+=1
              n+=1

      atomicBasisSetGroup.create_dataset("NbBasisGroups",(1,),dtype="i4",data=n)
      

else :
  for x in range(NbSpecies):
      #MyIdx=idxSpeciestoAtoms[x][0]
      atomicBasisSetGroup=GroupBasisSet.create_group("atomicBasisSet"+str(x))
      mylen="S"+str(len(uniq_atoms[x][0]))
      
      strList=[uniq_atoms[x][0]]
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('elementType', (1,),mylen, asciiList)
      strList=['cartesian']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('angular', (1,),'S9', asciiList)
      
      strList=['Gamess']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('expandYlm', (1,),'S6', asciiList)
      
      atomicBasisSetGroup.create_dataset("grid_npts",(1,),dtype="i4",data=1001)
      atomicBasisSetGroup.create_dataset("grid_rf",(1,),dtype="i4",data=100)
      atomicBasisSetGroup.create_dataset("grid_ri",(1,),dtype="f8",data=1e-06)
      
      strList=['log']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('grid_type', (1,),'S3', asciiList)
      strList=['gaussian']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('name', (1,),'S8', asciiList)
      strList=['no']
      asciiList = [n.encode("ascii", "ignore") for n in strList]
      atomicBasisSetGroup.create_dataset('normalized', (1,),'S2', asciiList)
      
      
      #print ("l_basis_clean1 ",x, "  ", len(l_basis_clean[x].split()))
      mybasis=l_basis_clean[x].split()
      
      size_basis=len(mybasis)
      nshell=0
      coeff=[]
      exp=[]
      myID=mybasis[0] 
      i=1
      while True:
       if (i >= size_basis) :
          break  
       else : 
          l=d_l[mybasis[i]]
          i+=1
          shell_size=int(mybasis[i]);
          BasisGroup=atomicBasisSetGroup.create_group("basisGroup"+str(nshell))
          strList=['Gaussian']
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          BasisGroup.create_dataset('type',(1,),'S8',asciiList)
      
          mylen="S"+str(len((uniq_atoms[x][0]+str(nshell)+str(l))))
      
          strList=[uniq_atoms[x][0]+str(nshell)+str(l)] 
          asciiList = [n.encode("ascii", "ignore") for n in strList]
          BasisGroup.create_dataset('rid', (1,),mylen, asciiList)
      
          BasisGroup.create_dataset("NbRadFunc",(1,),dtype="i4",data=shell_size)
          Val_l=BasisGroup.create_dataset("l",(1,),dtype="i4",data=l)
          Val_n=BasisGroup.create_dataset("n",(1,),dtype="i4",data=nshell)
          RadGroup=BasisGroup.create_group("radfunctions")
          #print "<basisGroup",nshell," rid=",myID+str(nshell)+str(l)," n=",nshell,"  l=",l ,"NbRadFunc=",shell_size,"type=Gaussian>"
          k=0
          j=0
          n=0
          k=i+1
          for m in range(shell_size): 
              #print "k=",k," ",k+1,"  ",k+2
              DataRadGrp=RadGroup.create_group("DataRad"+str(m))
              DataRadGrp.create_dataset("exponent",(1,),dtype="f8",data=float(mybasis[k+1]))
              DataRadGrp.create_dataset("contraction",(1,),dtype="f8",data=float(mybasis[k+2]))
             #print  "<radfunc exponent=",mybasis[k+1]," contraction=",mybasis[k+2], "DataRad=",nshell,"IdRad=",m,"/>"
              k=k+3 
          i=i+3*(shell_size)+1;     
          nshell+=1     
      atomicBasisSetGroup.create_dataset("NbBasisGroups",(1,),dtype="i4",data=nshell)


Restricted=True
GroupParameter.create_dataset("IsComplex",(1,),dtype="b1",data=Complex)
GroupParameter.create_dataset("SpinRestricted",(1,),dtype="b1",data=Restricted)

GroupDet=H5_qmcpack.create_group("Super_Twist")
NbMO=mo_num
NbAO=ao_num

if not PBC:
  eigenset=GroupDet.create_dataset("eigenset_0",(NbMO,NbAO),dtype="f8",data=orderd_mo_coeff)
else:
  GroupCell=H5_qmcpack.create_group("Cell")
  GroupCell.create_dataset("LatticeVectors",(3,3),dtype="f8",data=ezfio.get_qmcpack_latticevectors())
  GroupDet.create_dataset("Coord",(1,3),dtype="f8",data=ezfio.get_qmcpack_supertwist())

  E_g, C_gamma, E_g_unsorted,C_gamma_unsorted =mo_k2gamma(qmc_phase,ezfio.get_qmcpack_mo_coef_reorder_idx_kpts())  
  
  eigenset=GroupDet.create_dataset("eigenset_0",(NbMO,NbAO),dtype="f8",data=C_gamma.real.T) 
  eigenset_imag=GroupDet.create_dataset("eigenset_0_imag",(NbMO,NbAO),dtype="f8",data=C_gamma.imag.T) 
  #WARNING, WE ARE NOT PASSING THE EIGENVAL BUT KPOINT ORDER!!! 
  eigenvalue=GroupDet.create_dataset("eigenval_0",(1,NbMO),dtype="f8",data=ezfio.get_qmcpack_mo_coef_reorder_idx_kpts())
 
  #Unsorted Mo_coeffs for Multideterminants order matching QP
  eigenset_unsorted=GroupDet.create_dataset("eigenset_unsorted_0",(NbMO,NbAO),dtype="f8",data=C_gamma_unsorted.real.T) 
  eigenset_unsorted_imag=GroupDet.create_dataset("eigenset_unsorted_0_imag",(NbMO,NbAO),dtype="f8",data=C_gamma_unsorted.imag.T) 

GroupParameter.create_dataset("numMO",(1,),dtype="i4",data=NbMO)
GroupParameter.create_dataset("numAO",(1,),dtype="i4",data=NbAO)



if Multidet:
 groupMultiDet=H5_qmcpack.create_group("MultiDet")
 groupMultiDet.create_dataset("NbDet",(1,),dtype="i4",data=n_det)
 if PBC :
    for i in range(nexcitedstate):
       myName="Coeff_"+str(i)
       groupMultiDet.create_dataset(myName,(n_det,),dtype="f8",data=psi_coef[:,i])
       myName="Coeff_"+str(i)+"_imag"
       groupMultiDet.create_dataset(myName,(n_det,),dtype="f8",data=psi_coef_imag[:,i])
 else:
    for i in range(nexcitedstate):
       myName="Coeff_"+str(i)
       groupMultiDet.create_dataset(myName,(n_det,),dtype="f8",data=psi_coef[i])

 groupMultiDet.create_dataset("nstate",(1,),dtype="i4",data=len(MyDetA))
 groupMultiDet.create_dataset("nexcitedstate",(1,),dtype="i4",data=nexcitedstate)
 groupMultiDet.create_dataset("Nbits",(1,),dtype="i4",data=len(det_a))
 
 groupMultiDet.create_dataset("CI_Alpha",(n_det,len(det_a)),dtype='i8',data=MultiDetAlpha)
 groupMultiDet.create_dataset("CI_Beta",(n_det,len(det_b)),dtype='i8',data=MultiDetBeta)
 

 print ('Wavefunction successfully saved to QMCPACK HDF5 Format')
 print (f'Use: "convert4qmc -orbitals  {title}.h5 -multidet {title}.h5" to generate QMCPACK')

else:

 print ('Wavefunction successfully saved to QMCPACK HDF5 Format')
 print ('Use: "convert4qmc -orbitals  {}.h5" to generate QMCPACK input files'.format(title))
 

H5_qmcpack.close()

#print ('If not saved to HDF5, you can generate H5 out from convert4qmc, if output of save_for_qmcpack directed to file ')
#print ('Use: "convert4qmc -QP  {} -hdf5" to generate QMCPACK input files'.format(title))
# Close the file before exiting



