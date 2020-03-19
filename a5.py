# attempt number 5. Rewriting everything for the fifth time now.
# hopefully will work now
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt
# import pickle
import time
start = time.time()
#-----------------------------------------------------------------------------
# Functions
##-----------------------------------------------------------------------------
def update_pre(neurons_ip,I_input,lastspike_pre):
	# print(().shape)
	increment = (-100*(neurons_ip[:] + 0.07)+(10/3)*I_input.reshape((n_input,1))) * TIME_STEP*1e-3
	neurons_ip += increment
	spike_flag_pre = SYNAP_PULSE_DURATION*np.ones(neurons_ip.shape)
	temp = (neurons_ip >= V_TH )
	neurons_ip[temp] = V_RESET
	lastspike_pre[temp] = 0
	spike_flag_pre[temp] = 0
	# print(I_input)
	# print(increment)
	return neurons_ip, spike_flag_pre, lastspike_pre, temp

def update_post(neurons_op,pre_spiking_log,weightMatrix,label,lastspike_post):
	syn_exc = exc_syn_current_model(pre_spiking_log) # write this
	syn_inh = inh_syn_current_model(post_spiking_log) # write this
	i1 =(I_BIAS__used*target).reshape(n_output,1)
	i2 = weightMatrix.T @ syn_exc.reshape((n_input,1))
	i3 = INHIBIT_WEIGHT_MATRIX @ syn_inh.reshape((n_output,1))
	# print(i1,i2,i3)
	increment_current = i1 + i2 + i3
	# increment_current = (I_BIAS__used*target).reshape(n_output,1) + weightMatrix.T@syn_exc.reshape((n_input,1)) + INHIBIT_WEIGHT_MATRIX@syn_inh.reshape((n_output,1))
	increment = (-100*(neurons_op[:] + 0.07)+(10/3)*increment_current[:]) * TIME_STEP * 1e-3
	neurons_op += increment
	spike_flag_post = SYNAP_PULSE_DURATION*np.ones(neurons_op.shape)
	temp = neurons_op >= V_TH
	neurons_op[temp] = V_RESET
	lastspike_post[temp] = 0 
	spike_flag_post[temp] = 0
	# print(j,i2.T+i3.T,i2.T+i3.T+i1.T,temp.T,target)
	# print(j,i2.T+i3.T+i1.T)
	# print(j,i2.T+i3.T,temp.T,target)
	return neurons_op, spike_flag_post, lastspike_post,temp

def exc_syn_current_model(pre_spiking_log):
	curr = SYNAP_CURR_CONS_EXC* (np.exp(pre_spiking_log/(-TAU_M_EXCI)) - np.exp(pre_spiking_log/(-TAU_S)))
	curr[pre_spiking_log >= SYNAP_PULSE_DURATION] = 0
	return np.sum(curr,axis = 1)

def inh_syn_current_model(post_spiking_log):
	curr = SYNAP_CURR_CONS_EXC* (np.exp(post_spiking_log/(-TAU_M_INH)) -np.exp(post_spiking_log/(-TAU_S)) )
	curr[post_spiking_log >= SYNAP_PULSE_DURATION] = 0
	return np.sum(curr,axis=1)

def fig_update(fig1,weightMatrix,wt_pdate,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op):
	
	# fig1, ((ax_wt, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
	fig1.clf()
	# fig1 = plt.figure(figsize=(20, 12))
	ax_pre = plt.subplot2grid((5, 3), (0, 0), colspan=3, fig=fig1)
	ax_post = plt.subplot2grid((5,3),(1,0),colspan=3,fig=fig1)
	ax_wt = plt.subplot2grid((5, 3), (2, 0), colspan=2, fig=fig1)
	ax_wt_up = plt.subplot2grid((5, 3), (2, 2), colspan=1, fig=fig1)
	ax_pre_time = plt.subplot2grid((5, 3), (3, 0), colspan=2,rowspan=1, fig=fig1)
	ax_post_time = plt.subplot2grid((5, 3), (3, 2), fig=fig1)
	ax_nin = plt.subplot2grid((5,3),(4,0),colspan=2,fig=fig1)
	ax_nout = plt.subplot2grid((5,3),(4,2),fig=fig1)
	# fig1, ((ax_wt, ax_pre), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
	plt.ion()

	x_ticks = np.arange(n_input)

	#plot for weightMatrix
	im_wt = ax_wt.imshow(weightMatrix.T)
	ax_wt.set_title("weightMatrix")
	ax_wt.set_xticks(x_ticks)
	# ax_wt.set_yticks(np.arange(n_output))
	ax_wt.set_yticks([])
	# ax_wt.set_xticklabels(np.arange(n_input))
	# ax_wt.set_yticklabels(np.arange(n_output))
	# plt.setp(ax_wt.get_xticklabels(), rotation=45, ha="right",	         rotation_mode="anchor")
	for i in range(n_input):
		for j in range(n_output):
			text = ax_wt.text(i,j,'%1.0f'%weightMatrix[i,j],ha='center',
			va='center',color='w')
	# cbar = ax_wt.figure.colorbar(im, ax=ax_wt , **cbar_kw)
	cbar0 = ax_wt.figure.colorbar(im_wt, ax=ax_wt)
		# plt.pause(0.5)

	#plot for prespikinglog
	im_pre = ax_pre.imshow(pre_spiking_log)
	ax_pre.set_title('Last spiking of pre neurons')
	# ax_pre.set_yticks(np.arange(n_input))
	ax_pre.set_yticks([])
	# cbar1 = ax_pre.figure.colorbar(im_pre, ax=ax_pre)
	# plt.pause(0.5)
	# plt.draw()

	#plot for postspiking log
	im_post = ax_post.imshow(post_spiking_log)
	ax_post.set_title('Last spiking of post neurons')
	ax_post.set_yticks([])
	# cbar6 = ax_post.figure.colorbar(im_post,ax=ax_post)


	#plot for weight updates
	im_wt_up = ax_wt_up.imshow(wt_update.T)
	ax_wt_up.set_title('weight updates')
	# ax_wt_up.set_yticks(np.arange(n_output))
	ax_wt_up.set_xticks(x_ticks)
	ax_wt_up.set_xticklabels(x_ticks)
	plt.setp(ax_wt_up.get_xticklabels(), rotation=45, ha="right",	         rotation_mode="anchor")
	ax_wt_up.set_yticks([])
	cbar2 = ax_wt_up.figure.colorbar(im_wt_up,ax=ax_wt_up)
	# plt.pause(0.5)

	# pre time
	im_pre_time = ax_pre_time.imshow(lastspike_pre.T)
	ax_pre_time.set_title('lastspike_pre')
	ax_pre_time.set_xticks(x_ticks)
	ax_pre_time.set_yticks([])
	for i in range(n_input):
		if lastspike_pre[i,0]>500:
			continue
		else:
			text = ax_pre_time.text(i,0,lastspike_pre[i,0],ha='center',
				va='center',color='w')
	# cbar2 = ax_pre_time.figure.colorbar(im_pre, ax=ax_pre_time)
	# plt.pause(0.5)

	#plot for post time
	im_post_time = ax_post_time.imshow(lastspike_post.T)
	ax_post_time.set_title('lastspike_post')
	ax_post_time.set_xticks([])
	ax_post_time.set_yticks([])
	for i in range(n_output):
		text = ax_post_time.text(i,0,lastspike_post[i,0],ha='center',
			va='center',color='w')
	cbar3 = ax_post_time.figure.colorbar(im_post_time, ax=ax_post_time)
	# plt.pause(0.5)


	#plot for input neurons
	ax_nin.set_title('neurons_ip')
	im_nin = ax_nin.imshow(neurons_ip.T)
	for i in range(n_input):
		text = ax_nin.text(i,0,'%.2f'%neurons_ip[i,0],ha='center',
			va='center',color='w')
	ax_nin.set_yticks([])
	cbar4 = ax_nin.figure.colorbar(im_nin,ax=ax_nin)
	ax_nin.set_xticks(x_ticks)
	# plt.pause(0.5)

	#plot for output neurons
	ax_nout.set_title('neurons_op')
	im_nout = ax_nout.imshow(neurons_op.T)
	ax_nout.set_yticks([])
	ax_nout.set_xticks(np.arange(n_output))
	for i in range(n_output):
		text = ax_nout.text(i,0,'%.2f'%neurons_op[i,0],ha='center',
			va='center',color='w')
	cbar5 = ax_nout.figure.colorbar(im_nout,ax=ax_nout)
	# plt.show()
	# plt.pause(0.5)
	return

def update_time():
	pass
	global pre_spiking_log
	global post_spiking_log
	pre_spiking_log += 1
	post_spiking_log += 1

	global lastspike_pre
	global lastspike_post
	lastspike_pre += 1
	lastspike_post += 1

	global time_index
	time_index += 1
	return

def update_weights(spike_flag_pre,spike_flag_post,lastspike_pre,lastspike_post,weightMatrix,wt_update):
	mask = np.zeros(weightMatrix.shape)
	temp1 = (lastspike_pre==0).squeeze()
	temp2 = (lastspike_post==0).squeeze()
	mask[temp1,:] = 1
	mask[:,temp2] = 1
	dt = np.zeros((n_input,n_output),dtype='int')
	for i in range(n_output):
		dt[:,i] -= lastspike_pre[:,0]
	for i in range(n_input):
		dt[i,:] += lastspike_post[:,0]

	# temp = (dt >=0).squeeze()
	# wt_update[temp] = A_UP*np.multiply( (1- weightMatrix/W_MAX),np.exp(-dt/TAU_UP) )
	# wt_update[not temp] = A_DOWN * np.multiply((weightMatrix/weightMatrix),np.exp(dt/TAU_DOWN))
	# wt_update[not mask] = 0
	# dt = [[-lastspike_pre[i]*3] for i in  range(n_input)]
	# dt = [(dt[i,:]+lastspike_post[:]) for i in range(n_input) ]
	for i in range(n_input):
		for j in range(n_output):
			if mask[i,j] == 0 :
				wt_update[i,j] = 0
			elif dt[i,j] >= 0 :
				# print(TAU_UP)
				wt_update[i,j] = A_UP*((1 - weightMatrix[i,j]/W_MAX)**1.7)*np.exp(-dt[i,j]/TAU_UP)
			else:
				wt_update[i,j] = A_DOWN*((weightMatrix[i,j]/W_MAX)**1.7)*np.exp(dt[i,j]/TAU_DOWN)
	weightMatrix += wt_update
	return weightMatrix, wt_update
#-----------------------------------------------------------------------------
#load data
#-----------------------------------------------------------------------------
print('loading data')
iris = datasets.load_iris()
X = iris.data[:,:]
Y = iris.target
n_output = 3 #number of target classes
#-----------------------------------------------------------------------------
# Data normalisation
#-----------------------------------------------------------------------------
print('Converting data set to required format')
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
print('Transforming data')
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
												random_state=13)
	#save transformed data in a .csv
#-----------------------------------------------------------------------------
# Constants 
#-----------------------------------------------------------------------------
print('Setting Constants')
	#time
# TIME_STEP = 1 #in mS
TIME_STEP = 0.1 #in mS
SINGE_SAMPLE_TIME =int(100.0/TIME_STEP)  #smillieconds/time_step
RESTING_TIME = int(50.0/TIME_STEP) #milliseconds/time_step
PULSE_TIME = int(50.0/TIME_STEP) #millisec
time_index = 0

	#neuron
V_TH = 0.02 #volts
V_RESET = 0 #volts
	#synapse
TAU_M_EXCI = int(10/TIME_STEP)
TAU_M_INH = int(50/TIME_STEP) #miliseconds
TAU_S = int(2.5/TIME_STEP) #mSec
SYNAP_CURR_CONS_EXC = 3e-4 #nA
SYNAP_CURR_CONS_INH = 4e-3 #nA
SYNAP_PULSE_DURATION = int ((50.0)/TIME_STEP) # mSec/time step

# I_BIAS = 2 #in nA
I_BIAS = 2 #in nA
# I_BIAS = 10
I_BIAS__used = I_BIAS

	#learning rule
TAU_UP = int(10/TIME_STEP)     #miliseconds
TAU_DOWN = int(20/TIME_STEP)   #miliseconds
A_UP = 10
A_DOWN = -7
W_MAX = 700

INHIBIT_WEIGHT_MATRIX = -400*np.ones((n_output,n_output),dtype='int')
np.fill_diagonal(INHIBIT_WEIGHT_MATRIX,0)
fig1 = plt.figure(figsize=(20, 12))
# plt.draw()
# plt.show()

print('Initialising Multi-use Variables')

	#synapse
pre_spiking_log = (SYNAP_PULSE_DURATION)*np.ones((n_input,SYNAP_PULSE_DURATION),dtype='L') # when pre fires;this is used to calculate the input current to the post neurons
post_spiking_log = (SYNAP_PULSE_DURATION)*np.ones((n_output,SYNAP_PULSE_DURATION),dtype='L')

	#neuron
#voltage of the capacitor on a LIF model neuron group
neurons_ip = np.zeros((n_input,1),dtype='float')	#present voltage
neurons_op = np.zeros((n_output,1),dtype='float')

spike_flag_pre	= np.zeros((n_input,1),dtype='int')
spike_flag_post	= np.zeros((n_output,1),dtype='int')

spike_countlog_pre = np.zeros((n_input,SYNAP_PULSE_DURATION),dtype='bool')
spike_countlog_post = np.zeros((n_output,SYNAP_PULSE_DURATION),dtype='bool')

lastspike_pre = (SYNAP_PULSE_DURATION)*np.ones((n_input,1),dtype='L')
lastspike_post = (SYNAP_PULSE_DURATION)*np.ones((n_output,1) ,dtype='L')
# lastspike_pre = lastspike_pre.squeeze()
# lastspike_post = lastspike_post.squeeze()

	#weight matrix
weightMatrix = np.random.normal(loc = 280, scale = 30, size = [n_input,n_output]) #random initialisation 
wt_update = np.zeros(weightMatrix.shape)

# I_bias = 1e-9 #nA
I_zero = np.zeros((n_input,1),dtype='float')

#-----------------------------------------------------------------------------
# Run the Simulations


# Training
print('Training')


# plt.tight_layout()
# plt.pause(0)
'''Add a dynamic plot showing pre,post spike log, accuracy averaged over a window of time '''
j = 0
epochs = 1
num_examples = epochs * x_train.shape[0]
time_index = 0
# I_input = []
# target = []
predictions = []
while j < num_examples :
	k=0
	# redo this indexing properly
	I_input = x_train[j%x_train.shape[0],:]
	target = y_train[j%x_train.shape[0],:]
	I_BIAS__used = I_BIAS
	neurons_ip *= 0
	neurons_op *= 0
	post_spiking_log += SYNAP_PULSE_DURATION
	pre_spiking_log += SYNAP_PULSE_DURATION
	while k < SINGE_SAMPLE_TIME :
		neurons_ip, spike_flag_pre, lastspike_pre, temp  = update_pre(neurons_ip,I_input,lastspike_pre)
		spike_countlog_pre[:,time_index%SYNAP_PULSE_DURATION] = temp[:,0]
		neurons_op, spike_flag_post, lastspike_post,temp = update_post(neurons_op,pre_spiking_log,weightMatrix,target,lastspike_post)
		spike_countlog_post[:,time_index%SYNAP_PULSE_DURATION] = temp[:,0]
		pre_spiking_log[:,time_index % SYNAP_PULSE_DURATION] = spike_flag_pre[:,0]
		post_spiking_log[:,time_index % SYNAP_PULSE_DURATION] = spike_flag_post[:,0]
		weightMatrix, wt_update = update_weights(spike_flag_pre,spike_flag_post,lastspike_pre,lastspike_post,weightMatrix,wt_update)
		# fig_update(fig1,weightMatrix,wt_update,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op)
		update_time()
		# plt.draw()
		# plt.pause(1e-15)
		# print(j,time_index,'train')
		k += 1
	# fig_update(fig1,weightMatrix,wt_update,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op)
	# plt.pause(1e-9)
	class_prob = np.sum(spike_countlog_post,axis=1)  #/ len(np.nonzero(np.any(spike_countlog_post != 0, axis=0)))
	# predicted_class = np.where(class_prob == np.amax(class_prob))
	predicted_class = (class_prob == np.amax(class_prob))
	if np.sum(predicted_class) == 1:
		predicted_class = np.argmax(predicted_class)
		predictions.append(predicted_class == np.argmax(target))
		train_accuracy = np.sum(predictions)/len(predictions)
	else:
		predictions.append(False)
		train_accuracy = np.sum(predictions)/len(predictions)

	print(np.argmax(target),class_prob,np.amax(class_prob),'\ttrain accuracy:',train_accuracy)
	# print(j,class_prob,predicted_class,np.argmax(target),predicted_class == np.argmax(target),'\t\ttrain accuracy :',train_accuracy)

	# fig_update(fig1,weightMatrix,wt_update,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op)
	# plt.pause(1e-9)
	j += 1
# fig_update(fig1,weightMatrix,wt_update,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op)
# plt.pause(0)
# plt.show()

print(train_accuracy)

# with open('trial2.pickle','wb') as f:
# 	pickle.dump(weightMatrix,f)

# Testing 
predictions =[]
num_examples = x_test.shape[0]
j = 0
I_BIAS__used = 0
while  j < num_examples:
	I_input = x_test[j,:]
	target = y_test[j,:]
	neurons_ip *=0
	neurons_op *= 0
	post_spiking_log += SYNAP_PULSE_DURATION
	pre_spiking_log += SYNAP_PULSE_DURATION
	k=0
	while k < SINGE_SAMPLE_TIME:
		pass
		neurons_ip, spike_flag_pre, lastspike_pre, temp  = update_pre(neurons_ip,I_input,lastspike_pre)
		spike_countlog_pre[:,time_index%SYNAP_PULSE_DURATION] = temp[:,0]
		neurons_op, spike_flag_post, lastspike_post,temp = update_post(neurons_op,pre_spiking_log,weightMatrix,target,lastspike_post)
		spike_countlog_post[:,time_index%SYNAP_PULSE_DURATION] = temp[:,0]
		pre_spiking_log[:,time_index % SYNAP_PULSE_DURATION] = spike_flag_pre[:,0]
		post_spiking_log[:,time_index % SYNAP_PULSE_DURATION] = spike_flag_post[:,0]
		update_time()
		k +=1
	class_prob = np.sum(spike_countlog_post,axis=1) # / np.nonzero(np.all(spike_countlog_post != 0, axis=0))[0]
	predicted_class = (class_prob == np.amax(class_prob))
	if np.sum(predicted_class) == 1:
		predicted_class = np.argmax(predicted_class)
		predictions.append(predicted_class == np.argmax(target))
		test_accuracy = np.sum(predictions)/len(predictions)
	else:
		predictions.append(False)
		test_accuracy = np.sum(predictions)/len(predictions)

	print(np.argmax(target),class_prob,np.amax(class_prob),'\ttest accuracy:',test_accuracy)
	
	# fig_update(fig1,weightMatrix,wt_update,pre_spiking_log,post_spiking_log,lastspike_pre,lastspike_post,neurons_ip,neurons_op)
	j+=1
print(train_accuracy)
print(test_accuracy)
# Save results

# Plot Results

end = time.time()
print(end-start)