# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:02:54 2016

@author: Ewen
"""

from __future__ import division
from scipy import *
from pylab import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D # librairie 3D

import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import *
#Run all the code, then modify the parameters from line 322 to line
#331 to test the simulation


#Stockage des noeuds et poids dans les tableaux correspondants
#Ici, le fichier texte contient un paragraphe avec un noeud par
#ligne puis une ligne vide, puis un paragraphe avec un poids
#par ligne

K=40
#Indiquez l'emplacement du fichier texte
path="C:\Users\Ewen\Documents\X2015\Info2016\ProjetMap\\noeud40.txt"
fichier=open(path,"r")
Noeud=[]
Poid=[]
compteur=0
ligne=fichier.readline()
while (compteur<K):
    Noeud=Noeud+[float(ligne)]#Stockage noeuds
    compteur=compteur+1
    ligne=fichier.readline()
ligne=fichier.readline()#Ligne vide
compteur=0
while (compteur<K):#Stockage poids
    Poid=Poid+[float(ligne)]
    compteur=compteur+1
    ligne=fichier.readline()

#Question 3

def fact(K):
    Fact=np.zeros(K+1)#Tableau de stockage de la factorielle
    Fact[0]=1
    Fact[1]=1
    for i in range(2,K+1):
        Fact[i]=Fact[i-1]*i
    return Fact

def question3(K,Noeud,Poid):
    Fact=fact(K)
    T=np.zeros((K,K))
    Ti=np.zeros((K,K))
    A=2**(K-1)*Fact[K]*np.sqrt(np.pi)
    An=np.zeros(K+1)
    for k in range(0,K+1):#Coefficients 
        An[k]=(Fact[k]*np.sqrt(np.pi)*2**k)**(-1/2)
    for i in range(0,K):    
        T[K-1][i]=np.sqrt(A/Poid[i])*(-1)**(K-1+i)/K #On calcule la valeur de la K-1 fct de Hermite en les Noeuds
        Ti[i][K-1]=An[K-1]*np.exp(-(Noeud[i]**2)/2)*T[K-1][i]
    for i in range(0,K):
        T[K-2][i]=Noeud[i]*T[K-1][i]/(K-1) #On se sert ici de la recurrence à 2 pas pour les Hk
        Ti[i][K-2]=An[K-2]*np.exp(-(Noeud[i]**2)/2)*T[K-2][i]
    for i in range(0,K):
        for k in range(K-3,-1,-1):
            T[k][i]=1/(k+1)*(Noeud[i]*T[k+1][i]-0.5*T[k+2][i]) # Même recurrence
            Ti[i][k]=An[k]*np.exp(-(Noeud[i]**2)/2)*T[k][i]
    for i in range(0,K):
        for k in range(0,K): # On rajoute au tableau des Hk[xi] les coefficients correspondants 
            T[i][k]=T[i][k]*An[i]*Poid[k]*np.exp((Noeud[k]**2)/2)
    return(T,Ti)
    
#Vérification
T,Ti=question3(K,Noeud,Poid)
Tm=np.mat(T)
Tim=np.mat(Ti)
M1=Tm*Tim
M2=Tim*Tm
#print(M1)
#print(M2)

#Question 4
def question4(T,Ti):
    Tm=np.mat(T)
    Tim=np.mat(Ti)
    T2m=np.kron(Tm,Tm)
    T2im=np.kron(Tim,Tim)
    return (T2m,T2im)

#Vérification
#T2m,T2im=question4(T,Ti)
#M1=T2m*T2im
#M2=T2im*T2m
#print(M1)
#print(M2)



#Question 5
Nt=1000
g=100
Temps=1
mu=0.5
ot=Temps/Nt
T,Ti=question3(K,Noeud,Poid)#Matrices nécessaires à la question 5
def question5(Nt,g,ot,mu,T,Ti):
    plt.clf()
    J=np.zeros(K,dtype=complex)
    for k in range(0,K):
        J[k]=-1j*(2*k+1-mu)+ot/2
    Hphi1=np.zeros(K) #Composante de phi1 dans la base de Hermite
    Hphi1[0]=1 #Initialisation
    phi0=np.dot(Ti,Hphi1)
    for n in range(0,Nt):
        Hphi1=Hphi1*np.exp(J) #phi1(tn+1/2) dans la base de Hermite
        phi2=np.dot(Ti,Hphi1)   #phi2(tn)
        phi2=np.exp(-1j*ot*g*abs(phi2)**2)*phi2 #phi2(tn+1)
        Hphi3=np.dot(T,phi2) #phi3(tn+1/2)
        Hphi1=Hphi3*np.exp(J) #Bouclage, phi(tn+1)
    phi1=np.dot(Ti,Hphi1)
    phi1=abs(phi1)
    plt.plot(Noeud,phi0,label="Fonction initiale")
    plt.plot(Noeud,phi1,label="question 5")
    plt.legend()
    plt.show()   
    norme=0  #Conservation de la norme L2
    for k in range(0,K):
        norme=norme+abs(Hphi1[k])**2
    norme=np.sqrt(norme)
    print("Norme Q5 =")
    print(norme)
    return phi1
    
#phi1=question5(Nt,g,ot,mu,T,Ti)
Hphi1=np.zeros(K)
Hphi1[0]=1
phi0=np.dot(Ti,Hphi1)
#testenergie(phi1,phi0,Noeud,mu,K)   


#Conservation de l'énergie
def testenergie(psi,psi0,Noeud,mu,K):
    sumpsi=0
    sumpsi0=0
    for i in range (0,K-1):
        sumpsi=sumpsi+abs((psi[i+1]-psi[i])/(Noeud[i+1]-Noeud[i]))**2+(Noeud[i]**2-mu)*abs(psi[i])**2+0.5*g*abs(psi[i])**4
        sumpsi0=sumpsi0+abs((psi0[i+1]-psi0[i])/(Noeud[i+1]-Noeud[i]))**2+(Noeud[i]**2-mu)*abs(psi0[i])**2+0.5*g*abs(psi0[i])**4
    sumpsi=sumpsi+abs((psi[i]-psi[i-1])/(Noeud[i]-Noeud[i-1]))**2+(Noeud[i]**2-mu)*abs(psi[i])**2+0.5*g*abs(psi[i])**4
    sumpsi0=sumpsi0+abs((psi0[i]-psi0[i-1])/(Noeud[i]-Noeud[i-1]))**2+(Noeud[i]**2-mu)*abs(psi0[i])**2+0.5*g*abs(psi0[i])**4
    print("sumpsi= ",sumpsi," ; ""sumpsi0= ",sumpsi0)
    
#testenergie(phi1,phi0,Noeud,mu,K)   
    
#Question 6

Borne=5 #Borne arbitraire pour l'espace [-10,10]
Nx=100 #Discrétisation spatiale
def question6(Borne,Nx,ot,mu,g,Nt):
    Phi=np.zeros(Nx)
    xX=np.linspace(-10,10,Nx)
    dx=Borne*2/Nx #Pas d'espace
    Rn=0 # Coefficient R obtenu par récurrence
    diag1=-1j*ot/2*np.ones(Nx-1) #Coeff apparaissant dans la matrice Mn
    diag2=-1j*ot/2*np.ones(Nx-1) #Coeff apparaissant dans la matrice Mn
    diag=(1j*ot/(dx)**2-1j*ot*mu+1)*np.ones(Nx)
    for i in range(0,Nx):# Implémentation de phi0, première fonction de Hermite
         Phi[i]=(np.pi)**(-1/2)*np.exp(-((-Borne+dx*i)**2)/2)
         Phi[i]=(np.pi)**(-1/2)*np.exp(-((-Borne+dx*i)**2)/2)
         diag[i]=diag[i]+((-Borne+dx*i)**2)*1j*ot/2
         Rn=Rn+Phi[i]**2 #Norme de phi0 au carré
    Phi0=Phi
    for t in range(0,Nt):
        Rn=-Rn
        for i in range(0,Nx):#Calcul de Rn+1/2
            Rn=Rn+2*Phi[i]**2
        Cn=1j*ot*Rn/2 #Coeff apparaissant dans la matrice Mn
        diag0=diag+Cn*np.ones(Nx) #Coeff apparaissant dans la matrice M
        diagonals = [diag0, diag1, diag2]
        Mn=sparse.diags(diagonals, [0, -1, 1])
        Mn=csc_matrix(Mn)
        invMn=splu(Mn)#Décomposition LU
        Phi=2*invMn.solve(Phi)-Phi#solve    
    Phi1=abs(Phi)
    plt.clf()
    plt.plot(xX,Phi1,label="Question 6")
    plt.plot(xX,Phi0,label="Fonction initiale")
    plt.legend()
    plt.show()   

#question6(Borne,Nx,ot,mu,g,Nt)

#Question 9



T,Ti=question3(K,Noeud,Poid)
T2m,T2im=question4(T,Ti)#Matrices nécessaires à la question 9

def matrice(ot,ex,ey,gam,Om,K,mu,g):#Calcul de la matrice
    Diag0=np.ones(K**2,dtype=complex)#de passage à 7 diagonales
    for k in range(0,K):
        for l in range(0,K):
            Diag0[k*K+l]=Diag0[k*K+l]-ot/(2*(1j-gam))*((2*(k+l+1)-mu)/2-ex/4*(2*k+1)-ey/4*(2*l+1))
    Diag11=np.zeros(K**2,dtype=complex)#k-1,l+1
    Diag12=np.zeros(K**2,dtype=complex)#k+1,l-1
    Diag1l=np.zeros(K**2,dtype=complex)#k,l-2
    Diag2l=np.zeros(K**2,dtype=complex)#k,l+2
    Diag1k=np.zeros(K**2-2*K,dtype=complex)#k-2,l
    Diag2k=np.zeros(K**2-2*K,dtype=complex)#k+2,l
    for k in range(0,K):
        for l in range(0,K):
            if (l==K-1):
                Diag11[k*K+l]=0
            else:
                Diag11[k*K+l]=-1j*ot/(2*(1j-gam))*Om/2*np.sqrt((l+1)*k)
            if (k==K-1):
                Diag12[k*K+l]=0
            else:
                Diag12[k*K+l]=1j*ot/(2*(1j-gam))*Om/2*np.sqrt(l*(k+1)) 
    Diag11=Diag11[K-1::]
    Diag12=Diag12[0:K**2-K+1]
    for k in range(0,K):
        for l in range(0,K):
            Diag1l[k*K+l]=-ot/(2*(1j-gam))*ey/4*np.sqrt(l*(l-1))
    Diag1l=Diag1l[2::]
    Diag2l=Diag1l
    for k in range(0,K-2):
        for l in range(0,K):
            Diag1k[k*K+l]=-ot/(2*(1j-gam))*ex/4*np.sqrt((k+2)*(k+1))
    Diag2k=Diag1k
    Diagonals = [Diag0,Diag11, Diag12, Diag1l,Diag2l,Diag1k,Diag2k]
    Mm=sparse.diags(Diagonals, [0,-(K-1), K-1,-2,2,-2*K,2*K])#Matrice de passage
    return Mm

#Impression du module de la fonction en 2d
def printmodule2d(ot,Nt,ex,ey,gam,Om,K,mu,g,T2m,T2im,NbrImage,Mc):
    plt.clf()
    compteur=0
    pasDePrint=int(Nt/NbrImage)
    Hpsi=np.zeros(K**2)
    Hpsi[0]=1
    x,y=Noeud,Noeud
    X,Y=meshgrid(x,y)
    for t in range(0,Nt):
        Sol=gmres(Mc,Hpsi)
        Hpsi=2*Sol[0]-Hpsi#psi1(tn+1/2) dans la base de Hermite
        Psi=array(np.dot(T2im,Hpsi))
        Psi=exp(ot*g/(1j-gam)*abs(Psi))*Psi#psi2(tn+1)
        Psi=Psi.reshape(K**2,)#Besoin de reshape pour faire fonctionner np.dot
        Hpsi=array(np.dot(T2m,Psi))#psi2(tn+1) dans la base de Hermite
        Hpsi=Hpsi.reshape(K**2,)#Besoin de reshape pour faire fonctionner gmres
        Sol=gmres(Mc,Hpsi)
        Hpsi=2*Sol[0]-Hpsi#psi3(tn+1) dans la base de Hermite
        normeL2=0    
        for k in range(0,K**2):
            normeL2=normeL2+abs(Hpsi[k])**2
        normeL2=np.sqrt(normeL2)
        Hpsi=Hpsi/normeL2 #Renormalisation
        compteur=compteur+1
        if (compteur>pasDePrint):#Plot
            print(normeL2) #Vérification norme=1
            compteur=0
            psi=np.dot(T2im,Hpsi)
            psi=abs(psi)
            Z=array(psi.reshape(K,K))
            figure=plt.figure()
            ax = plt.subplot(111)
            cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
            ax.clabel(cset, fontsize=9, inline=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

#Impression de la phase de la fonction en 2d
def printphase2d(ot,Nt,ex,ey,gam,Om,K,mu,g,T2m,T2im,NbrImage,Mc):
    plt.clf()
    compteur=0 #Compteur permettant d'imprimer à intervalle régulier
    pasDePrint=int(Nt/NbrImage) #Pas de l'impression
    Hpsi=np.zeros(K**2)
    Hpsi[0]=1
    x,y=Noeud,Noeud
    X,Y=meshgrid(x,y)
    for t in range(0,Nt):
        Sol=gmres(Mc,Hpsi)
        Hpsi=2*Sol[0]-Hpsi#psi1(tn+1/2) dans la base de Hermite
        Psi=array(np.dot(T2im,Hpsi))
        Psi=exp(ot*g/(1j-gam)*abs(Psi))*Psi#psi2(tn+1)
        Psi=Psi.reshape(K**2,)#Besoin de reshape pour faire fonctionner np.dot
        Hpsi=array(np.dot(T2m,Psi))#psi2(tn+1) dans la base de Hermite
        Hpsi=Hpsi.reshape(K**2,)#Besoin de reshape pour faire fonctionner gmres
        Sol=gmres(Mc,Hpsi)
        Hpsi=2*Sol[0]-Hpsi#psi3(tn+1) dans la base de Hermite
        normeL2=0    
        for k in range(0,K**2):#Calcul de la norme
            normeL2=normeL2+abs(Hpsi[k])**2
        normeL2=np.sqrt(normeL2)
        Hpsi=Hpsi/normeL2 #Renormalisation
        compteur=compteur+1
        if (compteur>pasDePrint):#Plot
            compteur=0
            psi=np.dot(T2im,Hpsi)
            psi=np.array(psi)
            psi=np.angle(psi)
            Z=array(psi.reshape(K,K))
            figure=plt.figure()
            ax = plt.subplot(111)
            cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
            ax.clabel(cset, fontsize=9, inline=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

Nt=500  #Time discretization
g=50    #Boson interaction factor
Temps=10  #Time of the simulation 
mu=10  #
ot=Temps/Nt #Time step
Om=1.7    #Rotating factor
gam=0.07  #Gradient decreasing factor
ex=0.5   #Energetic potential 
ey=0.5
NbrImage=10 #Number of images printed by printmodule2d

#After every change of the parameter, re-run the matrice function
#printmodule2d print the module of the wave function

Mc=matrice(ot,ex,ey,gam,Om,K,mu,g)
printmodule2d(ot,Nt,ex,ey,gam,Om,K,mu,g,T2m,T2im,NbrImage,Mc)


#printphase2d(ot,Nt,ex,ey,gam,Om,K,mu,g,T2m,T2im,NbrImage,Mc)
