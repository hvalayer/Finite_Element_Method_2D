import numpy as np
import matplotlib.pyplot as plt
import sys
from exampleFunctions import *


## Definition of the Reference Element
## Declaration of nodes, integration points, weights and shape functions

def defineRefElement(elementType, degree):
    
    class refElement:
        pass
    refElement.type = elementType    # 0 : quadrilateral, 1 : triangle
    refElement.degree = degree       # 1 : linear
    
    if elementType == 0 and degree == 1: # linear quadrilateral (Q4)
        # number of nodes and coordinates
        refElement.nOfElementNodes = 4
        refElement.nodesCoord = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        # integration points and weights (Gauss-Legendre quadrature)
        a = np.sqrt(1/3)
        z = np.array([[-a,-a], [a,-a], [a,a], [-a,a]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([1,1,1,1])
        refElement.nIP = 4
        xi = z[:,[0]]; eta = z[:,[1]]
        # shape functions
        refElement.N = 1/4*np.block([(xi-1)*(eta-1), -(xi+1)*(eta-1), (xi+1)*(eta+1), -(xi-1)*(eta+1)])
        # derivates of shape functions w.r.t xi and eta
        refElement.Nxi = 1/4*np.block([eta-1, -eta+1, eta+1, -eta-1])
        refElement.Neta = 1/4*np.block([xi-1, -xi-1, xi+1, -xi+1])
        
    elif elementType == 1 and degree == 1: # linear triangle (T3)
        # number of nodes and coordinates
        refElement.nOfElementNodes = 3
        refElement.nodesCoord = np.array([[0,0],[1,0],[0,1]])
        # integration points and weights (Gauss-Legendre quadrature)
        z = np.array([[2/3,1/6],[1/6,1/6],[1/6,2/3]])
        refElement.integrationPoints = z
        refElement.integrationWeights = np.array([1/6,1/6,1/6])
        refElement.nIP = 3
        xi = z[:,[0]]; eta = z[:,[1]]
        # shape functions
        refElement.N = np.block([1-xi-eta, xi, eta])
        # derivates of shape functions w.r.t xi and eta
        refElement.Nxi = np.block([-np.ones(np.shape(xi)), np.ones(np.shape(xi)), np.zeros(np.shape(xi))])
        refElement.Neta = np.block([-np.ones(np.shape(eta)), np.zeros(np.shape(eta)), np.ones(np.shape(eta))])
        
    else:
        print('Element not implemented yet')
        sys.exit()
    return refElement


## Mesh generation
## Computation of nodal matrix X and connectivity matrix T

def Mesh(dom,nx,ny,refElement): 
    elementType = refElement.type
    degree = refElement.degree
    
    x1 = dom[0]; x2 = dom[1]
    y1 = dom[2]; y2 = dom[3]
    
    npx = degree*nx+1
    npy = degree*ny+1
    npt = npx*npy
    x = np.linspace(x1,x2,npx)
    y = np.linspace(y1,y2,npy)
    x,y = np.meshgrid(x,y)
    x.shape = (npt,1)
    y.shape = (npt,1)
    X = np.block([x,y])

    if elementType == 0 and degree == 1: # linear quadrilateral
        T = np.zeros((nx*ny,4),dtype=int)
        for i in np.arange(ny):
            for j in np.arange(nx):
                ielem = i*nx + j
                inode = i*npx + j
                T[ielem,:] = [inode, inode+1, inode+npx+1, inode+npx]

    elif elementType == 1 and degree == 1: # linear triangle
        nx_2 = np.round(nx/2)
        ny_2 = np.round(ny/2)
        T = np.zeros((2*nx*ny,3),dtype=int)
        for i in np.arange(ny):
            for j in np.arange(nx):
                ielem = 2*(i*nx + j)
                inode = i*npx + j
                nodes = np.array([inode, inode+1, inode+npx+1,inode+npx])
                if (i<ny_2 and j <nx_2) or (i>ny_2-1 and j > nx_2-1):
                    T[ielem,:] = nodes[np.array([0,1,2])];
                    T[ielem+1,:] = nodes[np.array([0,2,3])];
                else:
                    T[ielem,:] = nodes[np.array([0,1,3])];
                    T[ielem+1,:] = nodes[np.array([1,2,3])];
    else:
        print('Element not implemented yet')
        sys.exit()
                    
    return X,T
    

# Auxiliary function to display the Mesh

def plotMesh(X,T,refElement):
    elementType = refElement.type
    if elementType == 0:        # quadrilateral element
        plotOrder = [0,1,2,3,0]
    elif elementType == 1:      # triangle element
        plotOrder = [0,1,2,0]
    plt.figure(figsize=(12,8))  # initialize figure
    for e in np.arange(T.shape[0]):
        Te = T[e,:]
        Xe = X[Te,:]
        plt.plot(Xe[plotOrder,0], Xe[plotOrder,1], 'k')
    plt.plot(X[:,0], X[:,1],'or')
    plt.title("Mesh")
    plt.show()


# Definition of Elemental stiffness matrix Ke and force vector Fe
    
def elementMatrices(Xe,T,refElement):
    # get information from reference element
    nOfElementNodes = refElement.nOfElementNodes 
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta 
    # initialization of matrix Ke and vector Fe
    Ke = np.zeros((nOfElementNodes,nOfElementNodes))
    fe = np.zeros(nOfElementNodes)
    # loop on integration points
    for ip in np.arange(nIP): 
        N_ip = N[ip,:]
        Nxi_ip = Nxi[ip,:] 
        Neta_ip = Neta[ip,:]
        J = np.array([ [ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
        dvolu = wIP[ip]*np.linalg.det(J)
        grad_ref = np.vstack((Nxi_ip, Neta_ip)) 
        grad = np.linalg.solve(J, grad_ref)
        Nx = grad[0,:]
        Ny = grad[1,:]
        # update values of Ke and Fe
        Ke = Ke + (np.outer(Nx,Nx) + np.outer(Ny,Ny))*dvolu
        x_ip = N_ip@Xe
        fe = fe + N_ip*sourceTerm(x_ip)*dvolu;
    return Ke,fe


# Solve the FEM linear system

def findSolution(X,T,K,f,dom):
    # store limits of the rectangular domain
    x1 = dom[0]; x2 = dom[1]
    y1 = dom[2]; y2 = dom[3]
    # identification of nodes on the borders with tolerance
    tol = 1e-4*(x2-x1)/np.sqrt(T.shape[0])
    nodes_x1 = np.where(abs(X[:,0]-x1)<tol)[0]
    nodes_x2 = np.where(abs(X[:,0]-x2)<tol)[0]
    nodes_y1 = np.where(abs(X[:,1]-y1)<tol)[0]
    nodes_y2 = np.where(abs(X[:,1]-y2)<tol)[0]
    # nodes on Dirichlet boundary
    nodesDir = np.unique(np.block([nodes_x1,nodes_x2,nodes_y1,nodes_y2]))
    # impose Dirichlet boundary condition using the exact solution function
    valDir = exactSol(X[nodesDir,:])[0]
    valDir.shape = (len(nodesDir),1)
    nOfNodes = X.shape[0]
    nodesUnk = np.setdiff1d(np.arange(nOfNodes),nodesDir)
    f_red = f - K[:,nodesDir]@valDir
    # system reduction and solving
    K_red = K[nodesUnk,:][:,nodesUnk]
    f_red = f_red[nodesUnk,:]
    sol = np.linalg.solve(K_red,f_red)
    # creation of solution vector u
    u = np.zeros((nOfNodes,1))
    u[nodesDir] = valDir
    u[nodesUnk] = sol
    return u



## L2 and H1 error computation

def computeError(u,X,T,refElement):
    nIP = refElement.nIP
    wIP = refElement.integrationWeights 
    N    = refElement.N
    Nxi  = refElement.Nxi 
    Neta = refElement.Neta
    
    errorL2num = 0
    errorL2den = 0
    errorH1num = 0
    errorH1den = 0
    
    # loop on elements
    for e in range(len(T)):
        Te = T[e,:]
        Xe = X[Te]
        Ue = u[Te]
        # loop on integration points
        for ip in np.arange(nIP):
            N_ip = N[ip,:]
            Nxi_ip = Nxi[ip,:] 
            Neta_ip = Neta[ip,:]
            J = np.array([[ Nxi_ip@Xe[:,0], Nxi_ip@Xe[:,1]], [Neta_ip@Xe[:,0], Neta_ip@Xe[:,1]]])
            dvolu = wIP[ip]*np.linalg.det(J)
            grad_ref = np.vstack((Nxi_ip, Neta_ip))
            grad = np.linalg.solve(J, grad_ref)
            [dux_h,duy_h] = np.dot(grad,Ue)
            x_ip = N_ip@Xe
            x_ip.shape = [1,2]
            u_ex, du_x, du_y = exactSol(x_ip)
            errorL2num += (u_ex - N_ip.dot(Ue))**2*dvolu
            errorL2den += u_ex**2*dvolu
            errorH1num += ((du_x - dux_h)**2+(du_y-duy_h)**2)*dvolu
            errorH1den += (du_x**2+du_y**2)*dvolu
    return np.sqrt(errorL2num/errorL2den)[0],np.sqrt(errorH1num/errorH1den)[0]



# L2 and H1 error convergence plot

def plotErrors(domain,elementType,degree):
    referenceElement = defineRefElement(elementType, degree)
    vect_n = np.array([4,8,16,32])
    vect_L2err = np.zeros(len(vect_n))
    vect_H1err = np.zeros(len(vect_n))
    for n in range(len(vect_n)):
        nx = ny = vect_n[n]
        X,T = Mesh(domain,nx,ny,referenceElement)

        # initialise system matrix and rhs vector
        nOfNodes = X.shape[0]
        nOfElements = T.shape[0]
        K = np.zeros((nOfNodes,nOfNodes))
        f = np.zeros((nOfNodes,1))
        # loop in elements to obtain the system of equations
        for e in np.arange(nOfElements):
            Te = T[e,:]
            Xe = X[Te,:]
            Ke,fe = elementMatrices(Xe,T,referenceElement)
            for i in np.arange(len(Te)):
                f[Te[i]] = f[Te[i]] + fe[i]
                for j in np.arange(len(Te)):
                    K[Te[i],Te[j]] = K[Te[i],Te[j]] + Ke[i,j]  
        
        # impose BC and solve the system
        u = findSolution(X,T,K,f,domain)
        # compute L2 and H1 errors
        vect_L2err[n],vect_H1err[n] = computeError(u,X,T,referenceElement)
    
    if elementType == 1:
        t_Elem = 'linear triangular'
    else:
        t_Elem = 'linear quadrilateral'
        
    logn = np.log10(vect_n)
    logL2 = np.log10(vect_L2err)
    logH1 = np.log10(vect_H1err)
    plt.figure(figsize=(12,8))
    plt.plot(logn,logL2,'bo-',label="L2 Error")
    plt.plot(logn,-(degree+1)*logn,'b--',label="O(n²)")
    plt.plot(logn,logH1,'ro-',label="H1 Error")
    plt.plot(logn,-degree*logn,'r--',label="O(n)")
    plt.xlabel("log(n)"), plt.ylabel("log(E)")
    plt.legend(loc="best"); plt.grid()
    plt.title("Convergence plot for %s elements" %t_Elem)
    plt.show()