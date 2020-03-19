import snn_funcn
import numpy as np
from sklearn import datasets
from scipy.stats import norm
from sklearn.model_selection import train_test_split

SYNAP_CURR_CONS_EXC_set =[5e-5,7e-5,1e-4,3e-4,5e-4,7e-4,1e-3,3e-3,5e-3,7e-3,1e-2] #nA
SYNAP_CURR_CONS_INH_set =[1e-4,3e-4,5e-4,7e-4,1e-3,3e-3,5e-3,7e-3,1e-2,3e-2,5e-2] #nA

A_UP_set = [5,10]
A_DOWN_set = [-3,-7]

I_BIAS_set = [0,1,1.5]
epochs_set = [1,5,10]


#-----------------------------------------------------------------------------
#load data
#-----------------------------------------------------------------------------
# print('loading data')
iris = datasets.load_iris()
X = iris.data[:,:]
Y = iris.target
n_output = 3 #number of target classes
#-----------------------------------------------------------------------------
# Data normalisation
#-----------------------------------------------------------------------------
# print('Converting data set to required format')
for i in range(X.shape[1]):
	X[:,i] = (X[:,i] - X[:,i].min())/((X[:,i].max() - X[:,i].min()))
# X = np.

temp = -1*np.ones((X.shape[0],n_output),dtype='int') #desired output matrix or error matrix
for i in range(Y.shape[0]):
	temp[i,Y[i]] = 1 
Y = temp
#-----------------------------------------------------------------------------
# Data transformation
#-----------------------------------------------------------------------------
# print('Transforming data')
numfeatures = X.shape[1]
transformation_sd = 0.2 #for split =4
I_max = 4 #value in nA
feature_split = 4 # number of neurons to be dedicated for each feature, depends on the transformation used

# num_examples = X.shape[0]
transformation_means = np.arange(0,1+1/feature_split,1/feature_split)
# print(transformation_means)
feature_split += 1
n_input = numfeatures * feature_split

temp = np.zeros((X.shape[0],n_input),dtype='float')
for i in range(X.shape[0]):
	for j in range(numfeatures):
		temp[i,feature_split*j:feature_split*(j+1)] = I_max * norm.pdf(X[i,j],transformation_means[:],transformation_sd)
X = temp
x_train,x_test,y_train,y_test = train_test_split(X,Y,
												train_size=0.2,
												test_size=0.8,
												random_state=113)
	#save transformed data in a .csv
print("I_BIAS, SYNAP_CURR_CONS_EXC, SYNAP_CURR_CONS_INH, A_UP, A_DOWN, epochs, train_accuracy, test_accuracy")

for I_BIAS in I_BIAS_set:
	for SYNAP_CURR_CONS_INH,SYNAP_CURR_CONS_EXC in zip(SYNAP_CURR_CONS_INH_set,SYNAP_CURR_CONS_EXC_set):
		for A_UP,A_DOWN in zip(A_UP_set,A_DOWN_set):
			for epochs in epochs_set:
				arg1 = [x_train,y_train,x_test,y_test,I_BIAS,SYNAP_CURR_CONS_EXC,SYNAP_CURR_CONS_INH,A_UP,A_DOWN,epochs,n_input,n_output]
				train_accuracy,test_accuracy = snn_funcn.train_model(arg1)

				print(I_BIAS,',',SYNAP_CURR_CONS_EXC,',',SYNAP_CURR_CONS_INH,',',A_UP,',',A_DOWN,',',epochs,',',train_accuracy,',',test_accuracy)
pass
