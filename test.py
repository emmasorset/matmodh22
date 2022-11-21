print('heihei') #feil

print('hei igjen')

import numpy as np

def laplacePolar(A,n,dr,dtheta):   #tenker A er en matrise
    f=np.zeros((n,n))
    f[0,0]=A[0,0]              #trenger en form for endebetingelser
    f[n-1,n-1]=A[n-1,n-1]      #her er jeg også usikker
    for i in range(1,n-2):
        for j in range(1,n-2):
            d2fdr2=(A[i+1,j]-2*A[i,j]+A[i-1,j])/(dr**2)
            dfdr=(A[i+1,j]-A[i-1,j])/(2*dr)
            d2fdtheta2=(A[i,j+1]-2*A[i,j]+A[i,j-1])/(dtheta**2)
            f[i,j]=d2fdr2 + (1/i)*dfdr + (1/i**2)*d2fdtheta2
    return f
