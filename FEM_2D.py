from femFunctions import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
%matplotlib


# Domain definition 
domain = np.array([0,1,0,1])

# reference element and computational mesh
elementType = 0
degree = 1
referenceElement = defineRefElement(elementType, degree)
nx = 10; ny = 10
X,T = Mesh(domain,nx,ny,referenceElement)
plotMesh(X,T,referenceElement)

# initialize system matrix and source vector
nOfNodes = X.shape[0]
nOfElements = T.shape[0]
K = np.zeros((nOfNodes,nOfNodes))
f = np.zeros((nOfNodes,1))
# loop in elements to perform the assembly of elemental matrices and vectors
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
L2,H1 = computeError(u,X,T,referenceElement)

# Nodal error
u_ex = exactSol(X)[0]
u_ex.shape = u.shape
nodalErr = max(abs(u - u_ex))[0]
print('Nodal error : ', nodalErr)

# Solution plot formatting
npx = nx*degree+1
npy = ny*degree+1
x_grid = np.reshape(X[:,0],(npy,npx))
y_grid = np.reshape(X[:,1],(npy,npx))
if degree == 1:
    u_grid = np.reshape(u,(ny+1,nx+1))

if elementType == 1:
    t_Elem = 'linear triangular'
else:
    t_Elem = 'linear quadrilateral'
    
# 1) contour plot
plt.figure(figsize=(12,8))
plt.contourf(x_grid, y_grid, u_grid, 15, cmap="coolwarm")
plt.colorbar()
plt.title('FEM solution with %i %s elements' %(nx*ny, t_Elem))
#plt.show()

# 2) surface
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection='3d'); ax.set_title('FEM solution with %i %s elements' %(nx*ny, t_Elem))
surf = ax.plot_surface(x_grid, y_grid, u_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
#plt.show()

# 3) L2 and H1 error convergence plot
plotErrors(domain,elementType,degree)
L2,H1 = computeError(u,X,T,referenceElement)
print("L2    error : ", L2)
print("H1    error : ", H1)