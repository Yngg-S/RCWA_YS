# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 05:58:14 2022
this code is tested on python 3.7
not commented well yet.

next goal of V2.0 is to add features for anisotropic materials



@author: Yangyang Sun
"""






import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg as LA
import scipy.linalg as SLA


    
def conv_mat(A,P,Q,R):
    dim_A = np.shape(A)
    if np.shape(dim_A)[0] == 2:
        A = A[:,:,np.newaxis]
        R = 1
    if np.shape(dim_A)[0] ==1:
        A = A[:,np.newaxis,np.newaxis]
        Q = 1
        R = 1
    nx, ny, nz = np.shape(A)

    NH = P*Q*R
    p = np.arange(-1*(P-1)/2,1*(P-1)/2+1)
    q = np.arange(-1*(Q-1)/2,1*(Q-1)/2+1)
    r = np.arange(-1*(R-1)/2,1*(R-1)/2+1)
    
    A_fft =  np.fft.fftshift(np.fft.fftn(A))/(nx*ny*nz)
    p0 = int(1+np.floor(nx/2)-1)
    q0 = int(1+np.floor(ny/2)-1)
    r0 = int(1+np.floor(nz/2)-1)
    
    conv_mat = np.zeros((NH,NH))+0j
    # only pick the P, Q, R orders 
    for prow in range(P):
        for qrow in range(Q):
            for rrow in range(R):
                row = rrow*Q*P + qrow*P + prow
                for pcol in range(P):
                    for qcol in range(Q):
                        for rcol in range(R):
                            col = rcol*Q*P + qcol*P + pcol
                            pfft = int(p[prow] - p[pcol])
                            qfft = int(q[qrow] - q[qcol])
                            rfft = int(r[rrow] - r[rcol])
                            
                            
                            conv_mat[row,col] = A_fft[p0+pfft, q0+qfft, r0+rfft]
        
    return conv_mat

def right_divide(A,B):
# calculate the matrix C = A * inv(B)
    C_T = LA.solve(np.transpose(B) , np.transpose(A) )
    return np.transpose(C_T)

def redheffer_star(S11A,S12A,S21A,S22A,S11B,S12B,S21B,S22B):
    m , n  = np.shape(S11A)
    I = np.eye(m,n)
    
    
    
    D  = right_divide(S12A , I - np.matmul(S11B,S22A))
    F  = right_divide(S21B , I - np.matmul(S22A,S11B))
    
    
    
    S11AB = S11A + np.matmul(np.matmul(D,S11B), S21A)
    S12AB = np.matmul(D, S12B)
    
    S21AB = np.matmul(F, S21A)
    S22AB = S22B + np.matmul(np.matmul(F,S22A), S12B)
    return(S11AB,S12AB,S21AB,S22AB)


############################################
# dashboard
############################################
def rcwa3d(device, source):
#source 
    N_layer = len(device['L'])
    n_ref = np.sqrt(device['er_ref']*device['ur_ref'])
    n_trn = np.sqrt(device['er_trn']*device['ur_trn'])
    # reciprocal vectors
    d = device['t1'] [0]*device['t2'][1] - device['t2'][0]*device['t1'] [1]
    T1 = 2*np.pi* np.array([+device['t2'][1]/d,-device['t2'][0]/d])
    T2 = 2*np.pi* np.array([-device['t1'] [1]/d,+device['t1'] [0]/d])
    #print([T1,T2]) 
    k0 = 2*np.pi/source['lam0']
    kinc = n_ref*np.array([np.cos(source['theta']) * np.sin(source['phi']), np.sin(source['theta']) * np.sin(source['phi']) , np.cos(source['phi'])])
    p = np.arange(-1*np.floor(device['NP']/2),np.floor(device['NP']/2)+1)
    q = np.arange(-1*np.floor(device['NQ']/2),np.floor(device['NQ']/2)+1)
    Q, P = np.meshgrid(q,p)
    
    Kx = kinc[0]-P*T1[0]/k0 - Q*T2[0]/k0
    Ky = kinc[1]-P*T1[1]/k0 - Q*T2[1]/k0
    
    Kz_ref = np.conjugate(np.sqrt(device['ur_ref']*device['er_ref'] -Kx**2 - Ky**2+0j)) # backward  propagation, remember to add a negative sign later
    Kz_trn = np.conjugate(np.sqrt(device['ur_trn']*device['er_trn'] -Kx**2 - Ky**2+0j)) # forward propagation
    
    Kx_mat  = np.diag(Kx.flatten('F')) 
    Ky_mat  = np.diag(Ky.flatten('F')) 
    
    Kz_ref_mat  = np.diag(Kz_ref.flatten('F')) 
    Kz_trn_mat  = np.diag(Kz_trn.flatten('F')) 
    
    # identity matrix and zero matrix
    I_mat = np.eye(device['NP']*device['NQ'])
    Z_mat = np.zeros((device['NP']*device['NQ'],device['NP']*device['NQ']))
    
    # gap medium, eign mode
    Kz0_mat =np.conjugate( np.sqrt(I_mat - np.matmul(Kx_mat,Kx_mat) - np.matmul(Ky_mat,Ky_mat) + 0j ))
    Q0 = np.block([[ np.matmul(Kx_mat,Ky_mat),
                    I_mat - np.matmul(Kx_mat,Kx_mat)], 
                    [np.matmul(Ky_mat,Ky_mat)-I_mat,
                     -1*np.matmul(Kx_mat,Ky_mat)]])
        
    W0 =np.block([[I_mat, Z_mat],[Z_mat, I_mat]])
    LAM0 = np.block([[1j * Kz0_mat, Z_mat], [Z_mat, 1j*Kz0_mat]])
    V0 = right_divide(Q0,LAM0)
    
    #initialize global scattering matrix
    
    S11_glob = np.zeros((2*device['NP']*device['NQ'],2*device['NP']*device['NQ']))+0j
    S12_glob = np.eye(2*device['NP']*device['NQ'])+0j
    S21_glob = np.eye(2*device['NP']*device['NQ'])+0j
    S22_glob = np.zeros((2*device['NP']*device['NQ'],2*device['NP']*device['NQ']))+0j
    
    
    # main loop for each layer of device
    
    for n_layer in range(N_layer):
        ur_conv_i =device['ur_conv_mat'][:,:,n_layer] 
        er_conv_i =device['er_conv_mat'][:,:,n_layer] 
        P = np.block([[np.matmul(right_divide(Kx_mat,er_conv_i), Ky_mat),
                       ur_conv_i - np.matmul(right_divide(Kx_mat,er_conv_i), Kx_mat)],
                    [np.matmul(right_divide(Ky_mat,er_conv_i), Ky_mat) - ur_conv_i,
                     -1*np.matmul(right_divide(Ky_mat,er_conv_i), Kx_mat)]])
        Q = np.block([[np.matmul(right_divide(Kx_mat,ur_conv_i), Ky_mat),
                       er_conv_i - np.matmul(right_divide(Kx_mat,ur_conv_i), Kx_mat)],
                    [np.matmul(right_divide(Ky_mat,ur_conv_i), Ky_mat) - er_conv_i,
                     -1*np.matmul(right_divide(Ky_mat,ur_conv_i), Kx_mat)]])
       # conpute eigen mode
        LAM_sq, W = LA.eig(np.matmul(P, Q))    
        LAM = np.diag(np.sqrt(LAM_sq))
    
        V = np.matmul(Q,right_divide(W,LAM))
            
        X = SLA.expm(-1*LAM*k0*device['L'][n_layer])
        
        # layer s matrix
        A = LA.solve(W,W0) + LA.solve(V,V0)
        B = LA.solve(W,W0) - LA.solve(V,V0)
        D = A - np.matmul(np.matmul(np.matmul(X, right_divide(B,A)),X), B)
        S11_i = LA.solve(D, (np.matmul(np.matmul(np.matmul(X,right_divide(B,A)),X),A) -B)  )
        S12_i = np.matmul(LA.solve(D,X), (A-np.matmul(right_divide(B,A),B)) )
        S21_i = S12_i
        S22_i = S11_i
        
        # update global matrix
        S11_glob,S12_glob,S21_glob,S22_glob = redheffer_star(S11_glob,S12_glob,S21_glob,S22_glob,S11_i,S12_i,S21_i,S22_i)
    
    # external region s matrix
    # reflection region
    Q = 1/device['ur_ref']*np.block([[np.matmul(Kx_mat, Ky_mat),
                            device['ur_ref']*device['er_ref']*I_mat - np.matmul(Kx_mat,Kx_mat)],
                            [np.matmul(Ky_mat,Ky_mat) - device['ur_ref']*device['er_ref']*I_mat,
                            -1*np.matmul(Ky_mat,Kx_mat)]])
    #
    W_ref =  np.block([[I_mat ,Z_mat ], [Z_mat ,I_mat ]])
    
    LAM = np.block([[1j*Kz_ref_mat ,Z_mat ], [Z_mat ,1j * Kz_ref_mat ]])
    V_ref = right_divide(Q,LAM)
    A = LA.solve(W0,W_ref) + LA.solve(V0,V_ref)
    B = LA.solve(W0,W_ref) - LA.solve(V0,V_ref)
    S11_ref = -1*LA.solve(A,B)
    S12_ref = 2*right_divide(np.eye(np.shape(A)[0]),A)
    S21_ref = 0.5*(A-np.matmul(right_divide(B,A),B))
    S22_ref = right_divide(B,A)
    
    S11_glob,S12_glob,S21_glob,S22_glob = redheffer_star(S11_ref,S12_ref,S21_ref,S22_ref, S11_glob,S12_glob,S21_glob,S22_glob)
    
    # transmision region
    
    Q = 1/device['ur_trn']*np.block([[np.matmul(Kx_mat, Ky_mat),
                            device['ur_trn']*device['er_trn']*I_mat - np.matmul(Kx_mat,Kx_mat)],
                            [np.matmul(Ky_mat,Ky_mat) - device['ur_trn']*device['er_trn']*I_mat,
                            -1*np.matmul(Ky_mat,Kx_mat)]])
    W_trn =  np.block([[I_mat ,Z_mat ], [Z_mat ,I_mat ]])
    
    LAM = np.block([[1j*Kz_trn_mat ,Z_mat ], [Z_mat ,1j * Kz_trn_mat ]])
    V_trn = right_divide(Q,LAM)
    A = LA.solve(W0,W_trn) + LA.solve(V0,V_trn)
    B = LA.solve(W0,W_trn) - LA.solve(V0,V_trn)
    
    S11_trn = right_divide(B,A)
    
    S12_trn = 0.5*(A-np.matmul(right_divide(B,A),B))
    
    S21_trn = 2*LA.inv(A)
    S22_trn = LA.solve(-A,B)
    
    S11_glob,S12_glob,S21_glob,S22_glob = redheffer_star( S11_glob,S12_glob,S21_glob,S22_glob,S11_trn,S12_trn,S21_trn,S22_trn)
    
    ############################################
    # source: 
    ############################################
    # polarization
    norm = np.array([0,0,1])
    if np.abs(source['phi'] <1e-5):
        te_vector = np.array([0,1,0]) # if normal incident, TE polarized is along y axis
    else:
        te_vector = np.cross(kinc,norm) / LA.norm(np.cross(kinc,norm))
    tm_vector = np.cross(te_vector,kinc) / LA.norm(np.cross(te_vector,kinc))
    
    e_pol = source['pte']  * te_vector +source['ptm'] *tm_vector / LA.norm(source['pte']  * te_vector +source['ptm'] *tm_vector)
    
    # k space source
    k_source = np.zeros((device['NQ']*device['NP'],1))
    p0 = np.floor(device['NP']/2)
    q0 = np.floor(device['NQ']/2)
    m0 = (q0)*device['NP'] +p0
    k_source[int(m0)] = 1
    
    e_source = np.concatenate((e_pol[0]*k_source , e_pol[1]*k_source))
    
    c_src = LA.solve(W_ref,e_source)
    
    
    # reflective field
    c_ref = np.matmul(S11_glob, c_src)
    e_ref = np.matmul(W_ref,c_ref)
    rx = e_ref[:device['NQ']*device['NQ']]
    ry = e_ref[device['NQ']*device['NP']:device['NQ']*device['NP']*2]
    rz = -1*  np.matmul(right_divide(Kx_mat,Kz_ref_mat), rx) - np.matmul(right_divide(Ky_mat,Kz_ref_mat), ry)
    
    # reflective field
    c_trn = np.matmul(S21_glob, c_src)
    e_trn = np.matmul(W_trn,c_trn)
    tx = e_trn[:device['NQ']*device['NP']]
    ty = e_trn[device['NQ']*device['NP']:device['NQ']*device['NP']*2]
    tz = -1*  np.matmul(right_divide(Kx_mat,Kz_trn_mat), tx) - np.matmul(right_divide(Ky_mat,Kz_trn_mat), ty)
    
    #efficiency
    DE_ref =  np.matmul(np.real(Kz_ref_mat/kinc[2]) ,   (np.abs(rx)**2 +np.abs(ry)**2 +np.abs(rz)**2))
    REF = np.sum(DE_ref)
    DE_trn =  np.matmul(device['ur_trn']/device['ur_trn'] *np.real(Kz_trn_mat/kinc[2]) ,   (np.abs(tx)**2 +np.abs(ty)**2 +np.abs(tz)**2))
    TRN = np.sum(DE_trn)
    
    CON = REF+TRN


#    print([REF,TRN])
#    print(CON)

    DAT={}
    DAT['DE_ref'] = DE_ref
    DAT['DE_trn'] = DE_trn
    DAT['REF'] = REF
    DAT['TRN'] = TRN
    
    
    return DAT
    
def compute_conv_mat(device,ER,UR): 
    N_layer = len(device['L'])
    ER_conv = np.zeros((device['NP']*device['NQ'], device['NP']*device['NQ'],N_layer))+0j
    UR_conv = np.zeros((device['NP']*device['NQ'], device['NP']*device['NQ'],N_layer))+0j
    
    for n_layer in range(N_layer):
        if device['homo_flag'][n_layer] ==0:
            ER_conv[:,:,n_layer] = conv_mat(ER[n_layer],device['NP'],device['NQ'],device['NR'])
            UR_conv[:,:,n_layer] = conv_mat(UR[n_layer],device['NP'],device['NQ'],device['NR'])
        else:
            # TBA: need to simplify the calculation for homogenious layers
            ER_conv[:,:,n_layer] = conv_mat(ER[n_layer],device['NP'],device['NQ'],device['NR'])
            UR_conv[:,:,n_layer] = conv_mat(UR[n_layer],device['NP'],device['NQ'],device['NR'])
    return ER_conv, UR_conv


def clean_matrix(M,th):
    M[np.abs(M)<th] = 0
    return M
    


###########################################################################################################
# dashboard
###########################################################################################################

micrometers = 1
nanometers = 1e-3 * micrometers
degrees = np.pi/180
n_air = 1
n_SiO = 1.4496
n_SiN = 1.936
n_fs = 1.51

# first layer parameters
a = 1150 * nanometers
r = 400 * nanometers
# height of the layers
h1 = 230 * nanometers
h2 = 345 * nanometers

N1 =2048
N2 =2048




source = {}
device = {}
source['lam0'] = 1540 * nanometers # wavelength
source['theta'] = 0 * degrees
source['phi'] = 0 * degrees
source['pte'] = 1   # define polarization
source['ptm'] = 0
    
device['er_ref'] = 1.0
device['ur_ref'] = 1.0
device['er_trn'] = n_fs**2
device['ur_trn'] = 1

device['t1'] = a*np.array([np.cos(-60 * degrees), np.sin(-60 * degrees)])
device['t2'] = a*np.array([np.cos(60 * degrees), np.sin(60 * degrees)])
device['NP'] = 5
device['NQ'] = 5
device['NR'] = 1





###########################################################################################################
# build device
###########################################################################################################
L = [] # list of height layers
ER = [] # list of epsilon layers
UR = [] # list of epsilon layers
homo_flag = [] # indicate whether this layer is homogenious

t1_span = np.linspace(-0.5, 0.5, N1, endpoint=True)
t2_span = np.linspace(-0.5, 0.5, N2, endpoint=True)
t2_SPAN, t1_SPAN = np.meshgrid(t2_span,t1_span)
X_obl   = t1_SPAN * device['t1'][0] + t2_SPAN * device['t2'][0]
Y_obl   = t1_SPAN * device['t1'][1] + t2_SPAN * device['t2'][1]

# define layer 1:  hexagonal
b       = a * np.sqrt(3)
R2_1    = X_obl**2 + (Y_obl-b/2)**2
wall_region = R2_1 >= r**2

R2_2    = X_obl**2 + (Y_obl+b/2)**2
wall_region = np.logical_and(wall_region, R2_2 >= r**2)

R2_3    = (X_obl-a/2)**2 + (Y_obl)**2
wall_region = np.logical_and(wall_region, R2_3 >= r**2)

R2_4    = (X_obl+a/2)**2 + (Y_obl)**2
wall_region = np.logical_and(wall_region, R2_4 >= r**2)*1

er_1    = n_air + (n_SiO**2 - n_air**2) * wall_region*1

homo_flag.append(0)
L.append(h1)
ER.append(er_1)
UR.append(np.ones((N1,N2)))
# define layer 2: homogenious
homo_flag.append(1)
L.append(h2)
ER.append(n_SiN**2 * np.ones((N1,N2)))
UR.append(np.ones((N1,N2)))


# compute convolution matrix
device['L'] = L
device['homo_flag'] = homo_flag
device['er_conv_mat'],device['ur_conv_mat'] = compute_conv_mat(device,ER,UR)






###########################################################################################################
# prepare for parameter loop
###########################################################################################################

N_lam0      = 200
lam0_start  = 1530 *nanometers
lam0_end    = 1550 *nanometers
lam_list = np.linspace(lam0_start,lam0_end, N_lam0, endpoint = True)
N_phi      = 500
phi_start  = -20 * degrees
phi_end    = 20 * degrees
phi_list = np.linspace(phi_start,phi_end, N_phi, endpoint = True)
REF = np.zeros((N_lam0,N_phi)) # total reflection
TRN = np.zeros((N_lam0,N_phi))
CON = np.zeros((N_lam0,N_phi))


###########################################################################################################
# Lets go!
###########################################################################################################
# start sweep! Go!
for n_lam0 in range(N_lam0):
    for n_phi in range(N_phi):
    #for n_lam0 in range(N_lam0):
        source['lam0'] = lam_list[n_lam0]
        source['phi'] = phi_list[n_phi]
        #source['theta'] = lam_list[n_lam0]
        DAT = rcwa3d(device, source)
        REF[n_lam0,n_phi] = DAT['REF']
        TRN[n_lam0,n_phi] = DAT['TRN']
        CON[n_lam0,n_phi] = DAT['TRN'] + DAT['REF']
        if np.abs(DAT['TRN'] + DAT['REF']-1) >1e-3:
            print('warning: total energy is not conserved at [lambda, phi angle] =  ' )
            print([lam_list[n_lam0],phi_list[n_phi]])


#%%
# plot the results            
plt.figure()
plt.pcolor(X_obl,Y_obl,er_1)
plt.title('unit cell')
            
plt.figure()
plt.imshow(np.rot90(REF,1),extent=[lam_list[0]*1000,lam_list[-1]*1000,phi_list[0]/np.pi*180,phi_list[-1]/np.pi*180])
plt.ylim([-15,15])
plt.title('reflection vs. angles and wavelegnths')
plt.gca().set_aspect(3)
plt.colorbar()


plt.figure()
plt.plot(lam_list,REF[:,int((N_phi-1)/2)]*100,'b-')
plt.xticks(ticks=[1.53,1.535,1.54,1.545,1.55], labels = ['1.53','1.535','1.54','1.545','1.55'])
plt.title('normal incident reflection')
plt.tight_layout()
