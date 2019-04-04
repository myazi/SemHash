import numpy as np
def normalize(X):
	[n,m] = X.shape
	##特征维度归一化
	#L2norm = np.sum(np.multiply(X,X),1,keepdims=True)
	#L2norm[L2norm == 0] =1
	#L2norm = np.sqrt(L2norm)
	#for i in range(n):
	#	X[i,:] = X[i,:] / L2norm[i,0]
	
	##样本级别归一化
	L2norm = np.sum(np.multiply(X,X),0,keepdims=True)
	L2norm[L2norm == 0] =1
	L2norm = np.sqrt(L2norm)
	for i in range(m):
		X[:,i] = X[:,i] / L2norm[0,i]
	
	return X
