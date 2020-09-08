# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:54:46 2019

@author: zzxxq
"""

# -*- coding: utf-8 -*-

from threading import Thread
from Queue import Queue
from numpy import *
import pandas as pd
import time as t_module
import datetime
from scipy.stats import norm
from scipy.stats import poisson
from numba import jit
import matplotlib.pyplot as plt
from scipy.optimize import minimize

### Define initial parameters and read-in data
ratios_def=False
date_index=['2020-01-0'+str(i) for i in range(1,10)]+['2020-01-'+str(i) for i in range(10,32)]
date_index=date_index+['2020-02-0'+str(i) for i in range(1,10)]+['2020-02-'+str(i) for i in range(10,30)]
date_index=date_index+['2020-03-0'+str(i) for i in range(1,8)]#+['2020-03-'+str(i) for i in range(10,14)]
print 'please set value for extend_coef'
extend_coef=float(input())
print 'please select method'
method=int(input())
print 'please select starting date'
starting_dates=['2020-01-23','2020-02-02','2020-02-10']
sd_index=[7,12,16]
sel=int(input())

which_moving_average=False
starting_date1=starting_dates[sel]#
starting_date2=starting_date1
sd1_index=sd_index[sel]#7,16,12
sd2_index=sd1_index
key_time1='2020-01-26'#+gamma_i for reg
key_time2='2020-01-30'#+theta_j for popu
death_coef=[0.0111139,0.9528403,-0.0002758,0.0004522]

popu=pd.ExcelFile('moving_average_population_flow.xlsx')
popu_data={it.replace('inter-flow_',''):array(popu.parse(it))[:,1:] for it in popu.sheet_names}

ssm=lambda a,b:a+b
base_flow=amax(array([popu_data[it] for it in date_index]),axis=0)*1.1#reduce(ssm,[popu_data[it] for it in date_index[5:19]])/14.
before_spring_flow=reduce(ssm,[popu_data[it] for it in date_index[19:22]])/3.
bound_matrix=[base_flow,amax(array([before_spring_flow,before_spring_flow.T]),axis=0)]
data=pd.ExcelFile('result8.xlsx')
sheets=data.sheet_names
sheets=sort(sheets)
#print sheets
incub=14
incub1=7
os_T=2
p_I=[]
r=[]
W=[]
p_B=[]
hidden_infect=[]
raw_data=[]
for i in range(27):
	item ='ad_matrix_'+str(2*i)
	W.append(array(data.parse(item))[:,1:])
	item='prob_'+str(2*i)
	p_I.append(array(data.parse(item))[:,2])
	p_B.append(array(data.parse(item))[:,1])
	item='recovery_'+str(2*i)
	r.append(array(data.parse(item))[0,1])
	item='hidden_data_'+str(2*i)
	hd=data.parse(item)
	hidden_infect.append(array(hd[[it for it in hd.columns if 'pre' in str(it)]]))
	item='raw_data_'+str(2*i)
	raw_data.append(str(data.parse(item).columns[1]).split(' ')[0].replace('/','-'))
print r
raw_data=raw_data[::-1]
cities=array(data.parse('recovery_52'))[:,0:1]
mainland=[i for i in range(cities.shape[0]) if cities[i,0] not in [u'香港',u'澳门',u'台湾','others']]

W=W[::-1]
beds=pd.read_excel('beds.xlsx')
beds.index=beds['city']
beds=beds.ix[cities[mainland,0]]
beds=array(beds)[:,1:]
death_data=pd.read_excel('death ratio.xlsx')
death_data.index=death_data['city']

death_data=death_data.ix[cities[mainland,0]]
print death_data.ix[cities[mainland[13],0]]
death_data=array(death_data)[:,1:].astype(float)#[:,:len(W)]
print where(isnan(death_data))
cities=cities[1:,:]
cshape=cities.shape[0]
death_data[isnan(death_data)]=0
im=[W[i][1:,1:] for i in range(len(W))]
p_I=p_I[::-1]
p_B=p_B[::-1]
r=r[::-1]
hidden_infect=hidden_infect[::-1]

pl=[]
for date in date_index:
	if (datetime.datetime.strptime(date,'%Y-%m-%d')-datetime.datetime.strptime(starting_date2,'%Y-%m-%d')).days>=0:
		
		print date,sum(bound_matrix[0]-popu_data[date]<0),where(bound_matrix[0]-popu_data[date]<0)
for i in range(len(W)):
	ei,ev=linalg.eig(W[i].astype(float))
	pl.append(r[i]-max(absolute(ei)))
	print max(absolute(ei)),r[i]
	
### Define tool functions

def moving_matrix(s,l):
	ma=[]
	for i in range(len(s)-l):
		ma.append(s[i:i+l])
	return array(ma)
	
def moving_average(data,prob,incub,d_index,starting=None,theta=None,):
	# generate moving average network matrix subject to probability prob
	matrix=zeros_like(data[d_index[0]])
	a=zeros(data[d_index[0]].shape[0])
	b=zeros(data[d_index[0]].shape[0])
	
	for s in range(len(d_index))[incub:]:
		
		dd=zeros_like(matrix)
		
		for i,k in enumerate(d_index[s-incub:s]):
			
			if starting is not None:
				if (datetime.datetime.strptime(k,'%Y-%m-%d')-datetime.datetime.strptime(starting,'%Y-%m-%d')).days>=0:
					
					da=theta[k]
					
				else:
					
					da=0.
			else:
				da=data[k].copy()		
			dd=dd+da*prob[i]
			
		a=a+dot(dd,ones(dd.shape[1]))
		b=b+dot(ones(dd.shape[0]),dd)
		matrix=matrix+dd
	matrix=matrix/float(len(data)-incub)
	
	return matrix
def reg(W,popu_data,incub1,incub,os_T,date_index,p_I):
	# regress the network matrix W against the population flow matrix
	x,y=[],[]
	
	for i in range(len(W)):
		if os_T*(i)+incub1+incub<=len(date_index):
			d_index=array(date_index)[len(date_index)-incub1-incub-os_T*i:len(date_index)-os_T*i]
		
		else:
			d_index=array(date_index)[:len(date_index)-os_T*i]
		x.append(moving_average(popu_data,p_I[-i-1],incub,d_index))
	x=x[::-1]
	start,end=2,3
	inm=array(im[start:end]).flatten()
	xx=array(x[start:end],dtype=float).flatten()
	
	xx=array([ones_like(xx),xx],dtype=float).T
	#print dot(xx.T,xx)
	coef=linalg.inv(dot(xx.T,xx)).dot(dot(xx.T,inm))
	sumsquare=lambda coef:reduce(ssm,[sum((coef[0]+coef[1]*x[i]-W[i][1:,1:])**2) for i in range(len(W))[start:end]])
	print 'R^2', coef,sumsquare(coef)/float(len(inm))/std(inm)**2#coef1,res.fun/sum(inm**2),
	residuals={}
	for i in range(len(W)):
		residuals[i]=W[i][1:,1:]-coef[1]*x[i]-coef[0]
		residuals[i]=append(W[i][0:1,1:],residuals[i],axis=0)
		residuals[i]=append(W[i][:,0:1],residuals[i],axis=1)
	
	

	
	return coef,residuals
key_time1='2020-01-26'#+gamma_i for reg
key_time2='2020-01-30'#+theta_j for popu
def reg2(W,residuals,popu_data,incub1,incub,os_T,date_index,p_I,get_residuals=False,coef=None,coco=None):
	# regress the network matrix W against population flow and the flow-in, flow-out measures
	xx=[]
	dd_index=[]
	for i in range(len(residuals)):
		if os_T*(i)+incub1+incub<=len(date_index):
			d_index=array(date_index)[len(date_index)-incub1-incub-os_T*i:len(date_index)-os_T*i]
		
		else:
			d_index=array(date_index)[:len(date_index)-os_T*i]
		dd_index.append(d_index)
		xx.append(moving_average(popu_data,p_I[-i-1],incub,d_index))
	xx=xx[::-1]
	dd_index=dd_index[::-1]
	def obj(gamma,theta):
		x,y=[],[]
		theta0=diag(theta)
		theta0={it:dot(popu_data[it],theta0) for it in popu_data.keys()}
		for i in range(len(residuals)):
			x.append(moving_average(popu_data,p_I[i],incub,dd_index[i],key_time2,theta0))
		
		res=0
		
		gamma0=outer(ones_like(gamma),gamma)
		for i in range(len(residuals)):
			if (datetime.datetime.strptime(raw_data[i],'%Y-%m-%d')-datetime.datetime.strptime(key_time1,'%Y-%m-%d')).days<0:

				res+=sum((residuals[i][1:,1:]-x[i])**2)
				
			else:
				res+=sum((residuals[i][1:,1:]-x[i]-gamma0)**2)
		return res/float(len(residuals)*len(gamma)*len(theta))
	if not get_residuals:
		res=minimizer(lambda x:obj(x[:cshape],x[cshape:]),lambda x:True,x0=zeros(2*cshape),dt=0.00005,upper=0,lower=-0.1)
		coef1=res[0]
		print 'R^2', coef1[:cshape],coef1[cshape:],res[1]/obj(zeros(cshape),zeros(cshape))#std(array([residuals[i][1:,1:] for i in residuals.keys()]))**2#coef1,res.fun/sum(inm**2),
	else:
		coef1=coef
	residual1={}
	x,y=[],[]
	theta0=diag(coef1[cshape:])
	theta0={it:dot(popu_data[it],theta0) for it in popu_data.keys()}
	for i in range(len(residuals)):
		x.append(moving_average(popu_data,p_I[i],incub,dd_index[i],key_time2,theta0))
	
	for i in range(len(residuals)):
		if (datetime.datetime.strptime(raw_data[i],'%Y-%m-%d')-datetime.datetime.strptime(key_time1,'%Y-%m-%d')).days<0:
			gamma1=zeros(cshape)
		else:
			gamma1=coef1[:cshape]
		residual1[i]=residuals[i][1:,1:]-x[i]-outer(ones(cshape),gamma1)
		residual1[i]=append(residuals[i][0:1,1:],residual1[i],axis=0)
		residual1[i]=append(residuals[i][:,0:1],residual1[i],axis=1)
	
	for i in range(len(residual1)):
		if (datetime.datetime.strptime(raw_data[i],'%Y-%m-%d')-datetime.datetime.strptime(key_time1,'%Y-%m-%d')).days<0:
			gamma1=zeros(cshape)
		else:
			gamma1=coef1[:cshape]
		inff=residual1[i][1:,1:]+x[i]+outer(ones(cshape),gamma1)
		
		inff=append(residuals[i][0:1,1:],inff,axis=0)
		inff=append(residuals[i][:,0:1],inff,axis=1)
		print 'here',i,amax(inff-residuals[i])
		if get_residuals:
			inff=coco[0]+coco[1]*xx[i]+inff[1:,1:]
			inff=append(W[i][0:1,1:],inff,axis=0)
			inff=append(W[i][:,0:1],inff,axis=1)
			print 'here1',i,amax(inff-W[i]),amax(residual1[i][1:,1:])
	
	return coef1,residual1
def counterfact_sim(starting_date1,starting_date2,coef,ratios,gamma,theta,popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=True):
	# generate counterfactual network matrix W using the regression equation and the counterfactual set-ups (ratios)
	ratios=ratios.reshape(bound_matrix[0].shape[0],bound_matrix[0].shape[1])
	popu=popu_data.copy()
	popu1={}
	for date in date_index:
		if (datetime.datetime.strptime(date,'%Y-%m-%d')-datetime.datetime.strptime(starting_date2,'%Y-%m-%d')).days>=0:
			#print ratios
			
				
			popu1[date]=array(ratios*(bound_matrix[0]-popu[date])+popu[date])*(ratios>=0)+popu[date]*(1+ratios)*(ratios<0)#array(ratios*(popu[date])+popu[date])
			#print 1,amax(absolute(popu1[date]-popu[date])),amax(absolute(ratios*(bound_matrix[0]-popu[date])))
		elif (datetime.datetime.strptime(date,'%Y-%m-%d')-datetime.datetime.strptime(starting_date1,'%Y-%m-%d')).days>=0:
			#print ratios*(bound_matrix[1]-popu[date])
			
			popu1[date]=array(ratios*(bound_matrix[1]-popu[date])+popu[date])*(ratios>=0)+popu[date]*(1+ratios)*(ratios<0)
			#print 2,amax(absolute(popu1[date]-popu[date])),amax(absolute(ratios*(bound_matrix[0]-popu[date])))
		else:
			
			popu1[date]=popu[date]#array(0.*(bound_matrix[1]-popu[date])+popu[date])
			#print 3,amax(absolute(popu1[date]-popu[date])),amax(absolute(ratios*(bound_matrix[0]-popu[date])))
	x=[]
	gamma0=outer(ones_like(gamma),gamma)
	theta0=diag(theta)
	theta0={it:dot(popu1[it],theta0) for it in popu1.keys()}
	for i in range(len(W)):
		if os_T*(i)+incub1+incub<=len(date_index):
			d_index=array(date_index)[len(date_index)-incub1-incub-os_T*i:len(date_index)-os_T*i]
		
		else:
			d_index=array(date_index)[:len(date_index)-os_T*i]
		a=moving_average(popu,p_I[-i-1],incub,d_index)
		b=moving_average(popu1,p_I[-i-1],incub,d_index)
		b1=moving_average(popu1,p_I[-i-1],incub,d_index,key_time2,theta0)
		if (datetime.datetime.strptime(raw_data[-i-1],'%Y-%m-%d')-datetime.datetime.strptime(key_time1,'%Y-%m-%d')).days<0:

			y=a*coef[1]+(b-a)*coef[1]*extend_coef+coef[0]+b1
			
		else:
			y=a*coef[1]+(b-a)*coef[1]*extend_coef+coef[0]+b1+gamma0
		
		#y1=moving_average(popu,p_I[-i-1],incub,d_index)*coef
		#print 'shabi',amax(absolute(y-y1))
		y=append(zeros((1,y.shape[1])),y,axis=0)
		y=append(zeros((y.shape[0],1)),y,axis=1)
		
		y=y+residuals[len(W)-i-1]
		
		x.append(y)
		#print i,amax(y-W[-i-1]),amax(residuals[len(W)-i-1][1:,1:]),amax(b1-b),amax(gamma0),amax(b-a),amax(a*coef[1]+(b-a)*coef[1]*extend_coef+coef[0]+b1+gamma0+residuals[len(W)-i-1][1:,1:]-W[-i-1][1:,1:])
	x=x[::-1]	
	return x
	
	
def OS_forecast(data,p_I,net,r,incub,os_T):

	delta=data
	out=[]
	for i in range(os_T):
		o1=delta[:,-1]+dot(dot(net,delta[:,-incub:]),p_I)-r*delta[:,-1]
		delta=append(delta,o1.reshape(len(o1),1),axis=1)
		
	return delta[:,-os_T:]

def minimizer(obj,cons,x0,maxiter=100,upper=1,lower=0,dt=0.02):
	# a greedy searching algorithm
	dim=x0.shape[0]
	dt=dt
	t=0
	f0=obj(x0)
	star=0
	x00=x0.copy()
	success=False
	print 'initialized',x0,f0
	while t/dim<maxiter:
		tt=t%dim
		x=x0.copy()
		descending=False
		if x0[tt]+dt<=upper:
			x[tt]=x0[tt]+dt
			if cons(x):
				f=obj(x)
				print f
				if f<=f0:
					print 'descent+',t/dim,f0,f,f-f0
					x0=x.copy()
					f0=f
					descending=True
					success=True
		if not descending:
			if x0[tt]-dt>=lower:
				x[tt]=x0[tt]-dt
				if cons(x):
					f=obj(x)
					print f
					if f<f0:
						print 'descent-',t/dim,f0,f,f-f0
						x0=x.copy()
						f0=f
						descending=True
						success=True
						
		if not descending:
			star+=1
		else:
			star=0
		t+=1
		if star>=dim:
			if success:
				break
			else:
				
				x0=random.rand(dim)*(upper-lower)+lower
				star=0
	if not success:
		x0=x00
	return [x0,f0]
						
def minimizer1(obj,cons,x0,initialize=100,maxiter=100,upper=1,lower=0):
	# a greedy searching algorithm
	dim=x0.shape[0]
	dt=0.05
	t=0
	
	star=0
	x00=x0.copy()
	f0=obj(x0)
	x0=append(zeros(dim-2*cshape),0.0005*ones(2*cshape))#append(0.0005*ones(dim-2*cshape),zeros(2*cshape))
	
	initialized=False
	success=False
	while t<=initialize:
		con=cons(x0)
		f=obj(x0)
		if con and f<=f0:
			f0=f
			initialized=True
			success=True
			print 'successfully initialized'
			break
		elif not con:
			x0=upper*0.01-random.rand(dim)*(upper-lower)*0.01
		else:
			print 'partially successfully initialized'
			initialized=True
			break
		t+=1
	t=0
	
	while t/dim<maxiter:
		tt=t%dim
		x=x0.copy()
		descending=False
		if x0[tt]+dt<=upper:
			x[tt]=x0[tt]+dt
			if cons(x):
				f=obj(x)
				if f<=f0:
					print 'descent+',t/dim,f0,f,f-f0
					x0=x.copy()
					f0=f
					descending=True
					success=True
		if not descending:
			if x0[tt]-dt>=lower:
				x[tt]=x0[tt]-dt
				if cons(x):
					f=obj(x)
					if f<f0:
						print 'descent-',t/dim,f0,f,f-f0
						x0=x.copy()
						f0=f
						descending=True
						success=True
						
		if not descending:
			star+=1
		else:
			star=0
		t+=1
		if star>=dim:
			if success:
				break
			else:
				
				x0=random.rand(dim)*(upper-lower)+lower
				star=0
	if not success and not initialized:
		x0=x00
		f0=obj(x0)
		print 'fail to search better one'
	elif initialized and not success:
		print 'get feasible soultion:',x0,', f val:',obj(x0),', which is not better than the initial f:',f0
		f0=obj(x0)
	else:
		print 'successfully terminate'
	return [x0,f0]
			
				
					
		
def obj_func_r0(starting_date1,starting_date2,sd1_index,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,date_index):
	# Searching for Pareto optimal traffic restriction under the R0 constraint
	x0=W
	
	yy=[]
	for i,it in enumerate(x0):
		
		ei1,ev=linalg.eig(x0[i].astype(float))
		yy.append(r[i]-max(absolute(ei1)))
	def cons(ratios):
		gamma0,theta0=gamma*(1+ratios[-len(theta)-len(gamma):-len(theta)]),theta*(1+ratios[-len(theta):])
		
		ratios=ratios[:-len(theta)-len(gamma)]
		x=counterfact_sim(starting_date1,starting_date2,coef,ratios,gamma0,theta0,popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		#global x0
		#if type(x0) is not float:
		#	print 'shabi1', amax(absolute(array(x)-array(x0)))
		#x0=x
		y=[]
		for i,it in enumerate(x):
			#if i >=sd1_index:# and r[i]>0:
			#print sum(isnan(it.astype(float)))
			ei,ev=linalg.eig(it.astype(float))
				
			print i,amax(absolute(it-W[i])),round(r[i]-max(absolute(ei))-min([yy[i],0]),3)
			y.append(round(r[i]-max(absolute(ei))-min([yy[i],0]),2)>=0)
				
		#print absolute(ei)
		print 'fun',ratios,y
		y=all(y[sd1_index:])#min(y)#
		
		return y
	res=minimizer1(lambda x:-sum(x[x>=0]),cons,x0=zeros((residuals[0].shape[0]-1)*((residuals[0].shape[1]-1)+2)),maxiter=100,upper=1,lower=-1)#+0.01*sum(x[x<0]),maxiter=20)
	#minimize(obj_fun,x0=zeros((residuals[0].shape[0]-1)*(residuals[0].shape[1]-1)),bounds=[(0,1)]*(residuals[0].shape[0]-1)*(residuals[0].shape[1]-1),constraints={'type':'ineq','fun':cons})
	print res[1],res[0]#,obj_fun(zeros((residuals[0].shape[0]-1)*(residuals[0].shape[1]-1)))
	popu_matrix=counterfact_sim(starting_date1,starting_date2,coef,res[0][:(residuals[0].shape[0]-1)*(residuals[0].shape[0]-1)],res[0][-2*(residuals[0].shape[0]-1):-(residuals[0].shape[0]-1)],res[0][-(residuals[0].shape[0]-1):],popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		
	return res,popu_matrix
def obj_func_min_infect(starting_date1,starting_date2,sd1_index,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,hidden_infect,date_index):
	# Searching for Pareto optimal traffic restriction under the infection number constraint
	
	def obj_fun(ratios,null=False):
		gamma0,theta0=gamma*(1+ratios[-len(theta)-len(gamma):-len(theta)]),theta*(1+ratios[-len(theta):])
		
		ratios=ratios[:-len(theta)-len(gamma)]
		
		if null:
			x=W
		else:
			x=counterfact_sim(starting_date1,starting_date2,coef,ratios,gamma0,theta0,popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		#print 'here',amax(absolute(array(x)-array(W)))
		hidden_data=hidden_infect[1]#[sd1_index]
		k=1
		
		for i,matrix in enumerate(x):
			if i ==1:#sd1_index:
				
				hidden_data=append(hidden_data,OS_forecast(hidden_data,p_I[i],matrix,r[i],incub,incub1),axis=1)
				infect=[]
				for j in range(hidden_data.shape[0]):
					o2=dot(moving_matrix(hidden_data[j],incub),p_B[i])
					infect.append(o2)
				observed_infect=array(infect)
				#for j in range(observed_infect.shape[1]):
				#	if j>=observed_infect.shape[1]-incub1:
				#		
				#		death=death_data[:,-1:]*death_coef[2]+death_coef[3]*(observed_infect[:,j-8:j-7]-observed_infect[:,j-9:j-8])/observed_infect[:,j-9:j-8]+death_coef[4]+death_coef[0]*observed_infect[:,j:j+1]/beds
				#		death_data=append(death_data,death,axis=1)
				
		
				divisor0=ones_like(hidden_data)
				divisor1=ones_like(observed_infect)
		

		
		
			elif i>1:#sd1_index:
				starts=hidden_data.shape[1]-incub1+os_T
				if which_moving_average:
					hd=hidden_data/divisor0
				else:
					hd=hidden_data
				#print hd[:,:starts].shape
				y=append(hd[:,:starts],OS_forecast(hd[:,:starts],p_I[i],matrix,r[i],incub,incub1),axis=1)
				infect=[]
				for j in range(hidden_data.shape[0]):
					o2=dot(moving_matrix(y[j,-incub1-incub:],incub),p_B[i])
					infect.append(o2)
				infect=array(infect)
				if which_moving_average:
					hidden_data[:,hidden_data.shape[1]-incub1+os_T:]=hidden_data[:,hidden_data.shape[1]-incub1+os_T:]+y[:,y.shape[1]-incub1:y.shape[1]-os_T]
					hidden_data=append(hidden_data,y[:,-os_T:],axis=1)
		
					divisor0[:,divisor0.shape[1]-incub1+os_T:]=divisor0[:,divisor0.shape[1]-incub1+os_T:]+1
					divisor0=append(divisor0,ones_like(y[:,-os_T:]),axis=1)
				
					observed_infect[:,observed_infect.shape[1]-incub1+os_T:]=observed_infect[:,observed_infect.shape[1]-incub1+os_T:]+infect[:,infect.shape[1]-incub1:infect.shape[1]-os_T]
					observed_infect=append(observed_infect,infect[:,-os_T:],axis=1)
		
					divisor1[:,divisor1.shape[1]-incub1+os_T:]=divisor1[:,divisor1.shape[1]-incub1+os_T:]+1
					divisor1=append(divisor1,ones_like(infect[:,-os_T:]),axis=1)
				else:
					#hidden_data=(hidden_data*k+y[:,:y.shape[1]-os_T])/float(k+1)
					hidden_data=append(hidden_data[:,:-incub1+os_T],y[:,-incub1:],axis=1)
				
					#observed_infect[:,-incub1+os_T:]=(observed_infect[:,-incub1+os_T:]*k+infect[:,:-os_T])/float(k+1)
					observed_infect=append(observed_infect[:,:-incub1+os_T],infect[:,-incub1:],axis=1)
				
				
			k+=1
		if which_moving_average:
			observed_infect=observed_infect/divisor1
		return observed_infect
	x0=zeros((residuals[0].shape[0]-1)*((residuals[0].shape[1]-1)+2))
	original=obj_fun(x0,null=False)
	N=float(original[:,-os_T:].flatten().shape[0])
	ind=where(array(date_index[11:])==starting_date1)[0][0]
	def ob2(ratios):
		observed_infect=obj_fun(ratios)

		print method,'decay',amax(ratios[:-len(gamma)-len(theta)]),amin(ratios[:-len(gamma)-len(theta)]),mean(ratios),sum(observed_infect-original<0),amax(observed_infect-original),amax(original)
		return round(amax(observed_infect[:,ind:]-original[:,ind:]),0)<=0
		'''
		tester=any(observed_infect-original>0)
		if not tester:
			y=sum(observed_infect[:,-os_T:])/N
		else:
			y=sum(original[:,-os_T:])/N
		print 'fun',y
		return y
		'''
	def cons(ratios):
		return all([ratio>=0 for ratio in ratios])
	
		
	res=minimizer(lambda x:-sum(x[x>=0]),ob2,x0,maxiter=100,upper=1,lower=-1)#minimizer1(ob2,cons,x0=x0,maxiter=1000)#+0.01*sum(x[x<0])
	#minimize(obj_fun,x0=zeros((residuals[0].shape[0]-1)*(residuals[0].shape[1]-1)),bounds=[(0,1)]*(residuals[0].shape[0]-1)*(residuals[0].shape[1]-1))
	print res[1],res[0],original
	popu_matrix=counterfact_sim(starting_date1,starting_date2,coef,res[0][:(residuals[0].shape[0]-1)*(residuals[0].shape[0]-1)],res[0][-2*(residuals[0].shape[0]-1):-(residuals[0].shape[0]-1)],res[0][-(residuals[0].shape[0]-1):],popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		
	return res,popu_matrix

def obj_func_min_death(starting_date1,starting_date2,sd1_index,beds,death_coef,death_data,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,hidden_infect,date_index):
	# Searching for Pareto optimal traffic restriction under the death number constraint
	
	def obj_fun(ratios,death_data,null=False):
		#print 'shaboi',gamma.shape,theta.shape
		gamma0,theta0=gamma*(1+ratios[-len(theta)-len(gamma):-len(theta)]),theta*(1+ratios[-len(theta):])
		
		ratios=ratios[:-len(theta)-len(gamma)]
		
		if null:
			x=W
		else:
			x=counterfact_sim(starting_date1,starting_date2,coef,ratios,gamma0,theta0,popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		sd1=sd1_index-5
		hidden_data=hidden_infect[1]
		#calculate observed infections
		k=1
		for i,matrix in enumerate(x):
			if i ==1:
				
				hidden_data=append(hidden_data,OS_forecast(hidden_data,p_I[i],matrix,r[i],incub,incub1),axis=1)
				infect=[]
				for j in range(hidden_data.shape[0]):
					o2=dot(moving_matrix(hidden_data[j],incub),p_B[i])
					infect.append(o2)
				observed_infect=array(infect)
				#for j in range(observed_infect.shape[1]):
				#	if j>=observed_infect.shape[1]-incub1:
				#		
				#		death=death_data[:,-1:]*death_coef[2]+death_coef[3]*(observed_infect[:,j-8:j-7]-observed_infect[:,j-9:j-8])/observed_infect[:,j-9:j-8]+death_coef[4]+death_coef[0]*observed_infect[:,j:j+1]/beds
				#		death_data=append(death_data,death,axis=1)
				
		
				divisor0=ones_like(hidden_data)
				divisor1=ones_like(observed_infect)
		

		
		
			elif i>1:
				starts=hidden_data.shape[1]-incub1+os_T
				if which_moving_average:
					hd=hidden_data/divisor0
				else:
					hd=hidden_data
				y=append(hd[:,:starts],OS_forecast(hd[:,:starts],p_I[i],matrix,r[i],incub,incub1),axis=1)
				infect=[]
				for j in range(hidden_data.shape[0]):
					o2=dot(moving_matrix(y[j,-incub1-incub:],incub),p_B[i])
					infect.append(o2)
				infect=array(infect)
				if which_moving_average:
					hidden_data[:,hidden_data.shape[1]-incub1+os_T:]=hidden_data[:,hidden_data.shape[1]-incub1+os_T:]+y[:,y.shape[1]-incub1:y.shape[1]-os_T]
					hidden_data=append(hidden_data,y[:,-os_T:],axis=1)
		
					divisor0[:,divisor0.shape[1]-incub1+os_T:]=divisor0[:,divisor0.shape[1]-incub1+os_T:]+1
					divisor0=append(divisor0,ones_like(y[:,-os_T:]),axis=1)
				
					observed_infect[:,observed_infect.shape[1]-incub1+os_T:]=observed_infect[:,observed_infect.shape[1]-incub1+os_T:]+infect[:,infect.shape[1]-incub1:infect.shape[1]-os_T]
					observed_infect=append(observed_infect,infect[:,-os_T:],axis=1)
		
					divisor1[:,divisor1.shape[1]-incub1+os_T:]=divisor1[:,divisor1.shape[1]-incub1+os_T:]+1
					divisor1=append(divisor1,ones_like(infect[:,-os_T:]),axis=1)
				else:
					#hidden_data=(hidden_data*k+y[:,:y.shape[1]-os_T])/float(k+1)
					hidden_data=append(hidden_data[:,:-incub1+os_T],y[:,-incub1:],axis=1)
					#observed_infect[:,-incub1+os_T:]=(observed_infect[:,-incub1+os_T:]*k+infect[:,:-os_T])/float(k+1)
					observed_infect=append(observed_infect[:,:-incub1+os_T],infect[:,-incub1:],axis=1)
				
				
			k+=1
		if which_moving_average:
			observed_infect=observed_infect/divisor1
		#calculate death
		k=1
		ind=where(array(date_index[11:])==starting_date1)[0][0]
		#print death_data.shape,ind
		death_data=death_data[:,:ind].astype(float)
		#print death_data,where(isnan(death_data.astype(float)))
	
		for i in range(observed_infect.shape[1])[ind:]:
			hidden_data=death_data[:,death_data.shape[1]-1:]*death_coef[2]+death_coef[3]*(observed_infect[mainland,i-8:i-7]-observed_infect[mainland,i-9:i-8])/(observed_infect[mainland,i-9:i-8]+(observed_infect[mainland,i-9:i-8]==0))+death_coef[3]+death_coef[0]*observed_infect[mainland,i:i+1]/beds
			
			death_data=append(death_data,hidden_data,axis=1)
		#	print i,death_data[:,ind:].shape,observed_infect.shape
		death_data=death_data[:,ind:]*observed_infect[mainland,ind:]
		return death_data
	x0=zeros((residuals[0].shape[0]-1)*((residuals[0].shape[1]-1)+2))
	original=obj_fun(x0,death_data,null=True)
	N=float(original[:,-os_T:].flatten().shape[0])
	def ob2(ratios,death_data):
		death_data=obj_fun(ratios,death_data)
		y=(death_data-original).astype(float)
		print method,'decay',amax(ratios[:-len(gamma)-len(theta)]),amin(ratios[:-len(gamma)-len(theta)]),mean(ratios),amax(y),amax(original)#,death_data-original
		return all(round_(y,decimals=0)<=0)
		'''
		tester=any(death_data-original>0)
		if not tester:
			y=sum(death_data[:,-os_T:])/N
		else:
			y=sum(original[:,-os_T:])/N
		print 'fun',y
		return y
		'''
	def cons(ratios):
		return all([ratio>=0 for ratio in ratios])
	
	res=minimizer1(lambda x:-sum(x[x>=0]),lambda x:ob2(x,death_data),x0,maxiter=100,upper=1,lower=-1)#+0.01*sum(x[x<0])minimizer(ob2,cons,x0=x0,maxiter=20)
		
	#res=minimizer(lambda x:ob2(x,death_data),cons,x0=x0,maxiter=20)
	#minimize(obj_fun,x0=zeros((residuals[0].shape[0]-1)*(residuals[0].shape[1]-1)),bounds=[(0,1)]*(residuals[0].shape[0]-1)*(residuals[0].shape[1]-1))
	print res[1],res[0],original#obj_fun(zeros((residuals[0].shape[0]-1)*(residuals[0].shape[1]-1)),death_data)
	popu_matrix=counterfact_sim(starting_date1,starting_date2,coef,ratios,gamma0,theta0,popu_data,bound_matrix,residuals,p_I,incub1,incub,os_T,date_index,ratios_def=ratios_def)
		
	return res,popu_matrix

	
	
	
### run regression	
coef,residuals1=reg(W,popu_data,incub1,incub,os_T,date_index,p_I)
import os
if os.path.exists('gt.xlsx'):
	wr=pd.ExcelFile('gt.xlsx')
	coef1=array(wr.parse('gt'))[:,1:]
	gamma,theta=coef1[:,0],coef1[:,1]
	print 0,gamma.shape,theta.shape
	
	coef1,residuals=reg2(W,residuals1,popu_data,incub1,incub,os_T,date_index,p_I,get_residuals=True,coef=append(gamma,theta),coco=coef)#{int(it):array(wr.parse(it))[:,1:] for it in wr.sheet_names if it!='gt'}
else:
	coef1,residuals=reg2(W,residuals1,popu_data,incub1,incub,os_T,date_index,p_I)
	gamma,theta=coef1[:cshape],coef1[cshape:]
	gt=pd.DataFrame(append(cities,array([gamma,theta]).T,axis=1),columns=['city','gamma','theta'])
	wr=pd.ExcelWriter('gt.xlsx')
	gt.to_excel(wr,sheet_name='gt',index=False)
	cc=append(array([['others']]),cities,axis=0)
	for it in residuals.keys():
		d=pd.DataFrame(append(cc,residuals[it],axis=1),columns=['city','others']+list(cities.flatten()))
		d.to_excel(wr,sheet_name=str(it),index=False)
	wr.save()
print coef,gamma,theta


# Implementation of the counterfactual Pareto optimization
if method ==2:
	res2,popu2=obj_func_min_infect(starting_date1,starting_date2,sd1_index,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,hidden_infect,date_index)
	name='minimize_r0'
elif method==1:
	res2,popu2=obj_func_r0(starting_date1,starting_date2,sd1_index,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,date_index)
	name='minimize_infect'
else:
	res2,popu2=obj_func_min_death(starting_date1,starting_date2,sd1_index,beds,death_coef,death_data,coef,gamma,theta,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,hidden_infect,date_index)
	
	name='minimize_death'
writer2=pd.ExcelWriter('best_policies_csae'+str(method)+'_enlarge'+str(extend_coef)+'_starting_'+starting_date1+'_'+str(int(which_moving_average))+'.xlsx')
ratios2=pd.DataFrame(append(cities,res2[0][:-2*cshape].reshape(len(cities),len(cities)),axis=1),columns=['city']+list(cities.flatten()))	
gt=pd.DataFrame(append(cities,array([gamma,theta]).T*(1+res2[0][-2*cshape:].reshape(cshape,2)),axis=1),columns=['city','gamma','theta'])	

ratios2.to_excel(writer2,sheet_name=name,index=False)
gt.to_excel(writer2,sheet_name=name+'_gt',index=False)
#for k,item in enumerate(popu2):
#	out2=pd.DataFrame(append(cities,item,axis=1),columns=['city']+list(cities.flatten()))
#
#	out2.to_excel(writer2,sheet_name='minimize_infect_popu_'+str(k),index=False)

writer2.save()
'''
res1,popu1=obj_func_r0(starting_date1,starting_date2,sd1_index,coef,popu_data,bound_matrix,residuals,p_I,r,p_B,incub1,incub,os_T,date_index)
ratios1=pd.DataFrame(append(cities,res1[0].reshape(len(cities),len(cities)),axis=1),columns=['city']+list(cities.flatten()))
writer1=pd.ExcelWriter('best_policies_csae1.xlsx')
ratios1.to_excel(writer1,sheet_name='minimize_r0',index=False)

#for k,item in enumerate(popu1):
#	out1=pd.DataFrame(append(cities,item,axis=1),columns=['city']+list(cities.flatten()))
#
#	out1.to_excel(writer1,sheet_name='minimize_r0_popu_'+str(k),index=False)


writer1.save()

time_series=[]
population_residual={}
infection_residual={}
r={}
prob={}
for sheet in sheets:
	if 'res1' in sheet:
		time_series.append(sheet.replace('res1_',''))
		population_residual[time_series[-1]]=array(data.parse(sheet))[:,1:]
	elif 'res2' in sheet:
		infection_residual[sheet.replace('res2_','')]=array(data.parse(sheet))[:,1:]
	elif sheet=='base':
		base_flow=array(data.parse(sheet))[:,1:]
	elif sheet=='poly_popu_params':
		poly_popu_params=array(data.parse(it))
		poly_infect_params=poly_popu_params[:3]
		popu_infect_params=poly_popu_params[-1]
		poly_popu_params=poly_popu_params[3:-1]
	elif sheet=='ploy_ban':
		poly_ban=data.parse(sheet)
	
	elif sheet=='response_level':
		response_level=data.parse(sheet)
	elif sheet=='resume_prod':
		resume=data.parse(sheet)
	elif 'r' in sheet:
		r[sheet.replace('recovery_','')]=array(data.parse(sheet))[:,1:]
	elif 'prob' in sheet:
		prob[sheet.replace('prob_','')]=data.parse(sheet)
'''
