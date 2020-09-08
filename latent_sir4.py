

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


ssm=lambda a,b:a+b
def moving_matrix(s,l):
	#moving average in the time window [s-l,s]
	ma=[]
	for i in range(len(s)-l):
		ma.append(s[i:i+l])
	return array(ma)
		
def generate_true_infection(p_B,obs):
	# given observation and fixed p_B, generate the estimate to the hidden infection number 
	# via minimizing the L^2 difference between the obs and the estimates
	incub=len(p_B)
	infect_num=[]
	for i in range(obs.shape[0]):
		res=minimize(lambda x:sum((obs[i]-dot(moving_matrix(x,incub),p_B))**2),x0=append(obs[i],obs[i][-1]*ones(incub)),bounds=[(0,None)]*(obs.shape[1]+incub))
		#print res
		infect_num.append(res['x'])
	return array(infect_num),res['fun']
def _start(do_func,jobs,job_name,NUM):
	# NOTE: a parallel computing module, designed for the application of stochastic descending algorithm
	q = Queue()
	NUM = NUM
	def working():
		while True:
			arguments = q.get()
			start_time1=datetime.datetime.now()
			do_func(arguments)
			end_time1=datetime.datetime.now()
			delta1=end_time1-start_time1
			dh = int(delta1.seconds/3600)
			dm = int((delta1.seconds-dh*3600)/60)
			ds = delta1.seconds-dh*3600-dm*60
			delta1=job_name+' '+str(arguments)+' finished! '+str(dh)+'h:'+str(dm)+'m:'+str(ds)+'s spent'
			print delta1,datetime.datetime.now()
			t_module.sleep(0.1)
			q.task_done()
	#for number of tasks in quque
	print '-------',job_name,' started---------'
	start_time=datetime.datetime.now()
	for i in range(NUM):
		thread = Thread(target=working)
		thread.setDaemon(True)
		thread.start()
	#insert tasks to queue
	for s in jobs:
		q.put(s)
	#wait until all tasks done
	q.join()
	end_time=datetime.datetime.now()
	delta=end_time-start_time
	dh = int(delta.seconds/3600)
	dm = int((delta.seconds-dh*3600)/60)
	ds = delta.seconds-dh*3600-dm*60
	delta='-------',job_name,' finished in: '+str(dh)+'h:'+str(dm)+'m:'+str(ds)+'s:'
	print delta,datetime.datetime.now()

def gen_forecast(data,W,incub,p_I,p_B,hidden=False):
	# given the temporal network matrix W(t), infection prob p_I(t), and the past incub-day 
	# hidden infection number, generate the forecast of hidden infection number if hidden == True, 
	# generate the forecast of observed infection number (given the booming prob p_B(t)) if hidden == False
	delta=dot(W[0],data)#-dot(diag(W[1]),data)
	delta1=dot(diag(W[1]),data)
	out=[]
	for i in range(data.shape[0]):
		o1=append(data[i][:incub],data[i,incub-1]+cumsum(dot(moving_matrix(delta[i],incub),p_I)-delta1[i,incub-1:-1]))
		#o1=append(data[i][:incub],data[i,incub-1]+cumsum(dot(moving_matrix(delta[i],incub),p_I-delta1[i,incub-1:-1])))
		if not hidden:
			o2=dot(moving_matrix(o1,incub),p_B)
		else:
			o2=o1
		out.append(o2)
	return array(out)
def estimation(data,incub,wuhan_initial,p_B,wuhan_boom,date_index,fixed_initial=False,last_net=None):
	# back-forth estimate the temporal network matrix W(t), temporal infection prob p_I(t), 
	# and temporal recovery rate r(t) for fixed t and fixed hidden infection number, 
	# and fixed p_B(t)
	# data: hidden infection number
	# incub: length of incubation period
	# fixed_initial: hidden infection number estimated from the previous iteration in the back-propagation procedure
	# last_net: network matrix estimated from the previous iteration in the back-propagation procedure
	data1=data
	data=data[:,incub:]
	data1copy=data1.copy()
	datacopy=data.copy()
	
	if last_net is None:
		last_net=zeros((data.shape[0],data.shape[0]))
	global val,result
	val={}
	result=[]
	def do_func(i,prob,last_net):
		prob=prob[-incub-1:]
		obj_fun=lambda x,y:sqrt(sum( (datacopy[i][1:]-datacopy[i][:-1]-(dot(moving_matrix(dot(x,data1copy),incub),y[:-1])-y[-1]*data1copy[i,incub-1:-1])[:-1] )**2 )+sum((x-last_net[i])**2))#/10000.
		res=minimize(lambda x:obj_fun(x,prob),x0=zeros(data.shape[0]),bounds=[(0,1)]*(data.shape[0]))
		print  'res1',res['success'],res['fun']#,res['x']
		
		global val
		val[i]=res['x']
	global iprob
	iprob=append(wuhan_initial,append(ones(incub)/float(incub),zeros(1)))
	def change_initial(val,x,scalar=False):
		#print x.shape
		wuhan_initial=x[:-incub-1]
		x=x[-incub-1:]
		data2=data1.copy()
		data2[wuhan,:wuhan_initial.shape[0]]=wuhan_initial
		if not fixed_initial:
			matrix=array([val[i] for i in range(data.shape[0])])
			delta=data2[:,:incub]
			for i in range(data2.shape[1]-delta.shape[1]):
				#print x.shape,delta.shape
				o1=delta[:,-1]+dot(dot(matrix,delta[:,-incub:]),x[:-1])-dot(diag(x[-1]*ones(matrix.shape[0])),delta[:,-1])
				o1[non_wuhan]=data2[non_wuhan,incub+i]
				delta=append(delta,o1.reshape(len(o1),1),axis=1)
			data2=delta#gen_forecast(data2,[matrix,x[-1]*ones(matrix.shape[0])],incub,x[:-1],None,True)
			
		
			
		data3=data2[:,incub:]
		
		if scalar:
			y=reduce(ssm,[sum( (data3[i][1:]-data3[i][:-1]-(dot(moving_matrix(dot(val[i],data2),incub),x[:-1])-x[-1]*data2[i,incub-1:-1])[:-1] )**2 ) for i in range(data.shape[0])])/float(lc*incub1)**2# if i!=wuhan])
			if last_prob is not None:
				y=y+sum((x-last_prob)**2)
			return y
		else:
			y=reduce(ssm,[sum( (data3[i][1:]-data3[i][:-1]-(dot(moving_matrix(dot(val[i],data2),incub),x[:-1])-x[-1]*data2[i,incub-1:-1])[:-1] )**2 ) for i in range(data.shape[0])])/float(lc*incub1)**2# if i!=wuhan])
			if last_prob is not None:
				y=y+sum((x-last_prob)**2)
			return [y,data3,data2]
	
	def ob2(val):
		
		
		def obj_fun(x):
			y=change_initial(val,x)#,True)
			#return y[0]
			
			y1=dot(moving_matrix(y[-1][wuhan],incub),p_B)-wuhan_boom[wuhan]
			if datetime.datetime.strptime('2020-02-11','%Y-%m-%d') in date_index:
				change_point=where(date_index==datetime.datetime.strptime('2020-02-11','%Y-%m-%d'))[0][0]
			elif all([(datetime.datetime.strptime('2020-02-12','%Y-%m-%d')-date).days>0 for date in date_index]):  change_point=len(y1)
			else: change_point=0
			#print 'change point',change_point
			y1[:change_point]=y1[:change_point]*(y1[:change_point]<0).astype(float)
	
			
			return (y[0]+sum(y1**2))
			
			
		
		res=minimize(obj_fun,x0=iprob,bounds=[(0,None)]*(len(wuhan_initial)+incub+1),constraints={'type':'eq','fun':lambda x:sum(x[-incub-1:-1])-1})
		print 'res2',t,res#['success'],res['fun']
		return res#['x']
	t=1
	
	while t<maxiter:
		_start(lambda arguments:do_func(arguments,iprob,last_net),range(data.shape[0]),'iter',5)
		#print change_initial(val,iprob)
		iprob2=ob2(val)
		#print iprob2['x'].shape
		rp=False
		if t==1:
			copies=change_initial(val,iprob2['x'])
			data1copy=copies[-1]
			datacopy=copies[-2]
			result=[iprob2['fun'],{it:array([item for item in val[it]]) for it in val.keys()},iprob2['x'],datacopy,data1copy]
			
		else:
			if iprob2['fun']<result[0]:
				copies=change_initial(val,iprob2['x'])
				data1copy=copies[-1]
				datacopy=copies[-2]
				result=[iprob2['fun'],{it:array([item for item in val[it]]) for it in val.keys()},iprob2['x'],datacopy,data1copy]
				
		iprob=iprob2['x']
				
		if t>1:
			stop_condition=(reduce(ssm,[sum(absolute(val1[it]-val[it])) for it in val.keys()])+sum(absolute(iprob1-iprob)))/min([reduce(ssm,[sum(absolute(val1[it])) for it in val.keys()])+sum(absolute(iprob1)),reduce(ssm,[sum(absolute(val[it])) for it in val.keys()])+sum(absolute(iprob))])
			print t,stop_condition			
			if stop_condition<0.001:
				if iprob2['fun']>result[0]:
					iprob=result[2]
					val={it:array([item for item in result[1][it]]) for it in result[1].keys()}
					datacopy=result[3]
					data1copy=result[4]
				break
		val1={it:array([item for item in val[it]]) for it in val.keys()}
		iprob1=array([it for it in iprob])
				
		t+=1
	return array([val[it] for it in range(len(val))]),iprob[-1]*ones(data.shape[0]),iprob[-incub-1:-1],iprob[:-incub-1],datacopy,data1copy,iprob2['fun']

def sep_obj(data,wuhan,non_wuhan,data1,W,r,incub,p_I,p_B,date_index):
	# adjusted loss function given the special date '2020-02-11', see the
	# wuhan, non_wuhan: index for hubei province and the other provinces
	# data/data1:observed/hidden infection number
	if datetime.datetime.strptime('2020-02-11','%Y-%m-%d') in date_index:
		change_point=where(date_index==datetime.datetime.strptime('2020-02-11','%Y-%m-%d'))[0][0]
	elif all([(datetime.datetime.strptime('2020-02-11','%Y-%m-%d')-date).days>0 for date in date_index]): change_point=len(date_index)
	else: change_point=0	
	#print 'change point',change_point
	y=gen_forecast(data1,[W,r],incub,p_I,p_B)
	y=data-y
	y[:,:change_point]=y[:,:change_point]*(y[:,:change_point]>0).astype(float)
	
	y=sum(y**2)
	if last_boom is not None:
		y=y/float(lc*incub1)**2+sum((p_B-last_boom)**2)
	return y
def snd_step_est(data,incub,maxiter,os_T,name,date_index,pre_infer=None,last_net=None):
	# outer-loop: iteratively estimate p_B(t) and (p_I(t),W(t),recovery(t)) given the observation (data) and fixed t 
	# os_T: out-sample forecast horizon
	# date_index: a list of dates
	# incub: length of incubation period
	# name: label for saving output
	# pre_infer: hidden infection number estimated for the previous iteration in the back-propagation procedure
	# last_net: network matrix estimated for the previous iteration in the back-propagation procedure
	p_B=ones(incub)/float(incub)
	infer_data,fun0=generate_true_infection(p_B,data)
	fixed_initial=False
	if pre_infer is not None:
		infer_data[:,-pre_infer.shape[1]:]=pre_infer
		fixed_initial=True
		wuhan_initial=infer_data[wuhan,:-pre_infer.shape[1]]
	else:
		wuhan_initial=infer_data[wuhan,:incub]
	W,r,p_I,initial,data0,dd0,fun1=estimation(infer_data,incub,wuhan_initial,p_B,data,date_index,fixed_initial=fixed_initial,last_net=last_net)#data[wuhan]
	
	res=minimize(lambda prob:sep_obj(data,wuhan,non_wuhan,dd0,W,r,incub,p_I,prob,date_index),x0=p_B,bounds=[(0,None)]*(incub),constraints={'type':'eq','fun':lambda x:sum(x)-1})
	out=res['fun']
	p_B=res['x']
	out2=[out,p_B,p_I,W,r,dd0,data0]
	t=1
	while True:
		infer_data,fun01=generate_true_infection(p_B,data)
		if pre_infer is not None:
			infer_data[:,-pre_infer.shape[1]:]=pre_infer
		net1,recovery1,p_I1,initial,data1,dd1,fun11=estimation(infer_data,incub,initial,p_B,data,date_index,fixed_initial=fixed_initial,last_net=last_net)
		#print infer_data.shape,net1.shape,r.shape
		
		res=minimize(lambda prob:sep_obj(data,wuhan,non_wuhan,dd1,net1,recovery1,incub,p_I1,prob,date_index),x0=p_B,bounds=[(0,None)]*(incub),constraints={'type':'eq','fun':lambda x:sum(x)-1})
		out1=res['fun']
		
		p_B1=res['x']
		if out1<out2[0]:
			out2=[out1,p_B1,p_I1,net1,recovery1,dd1,data1]
		stop_condition=(sum(absolute(p_B1-p_B))+sum(absolute(p_I-p_I1))+sum(absolute(recovery1-r))+sum(absolute(net1-W)))/min([sum(absolute(p_B1))+sum(absolute(p_I1))+sum(absolute(recovery1))+sum(absolute(net1)),sum(absolute(p_B))+sum(absolute(p_I))+sum(absolute(r))+sum(absolute(W))])		
		print 'res3',name,t,out,out1,out2[0],fun1,fun11,stop_condition,res
		if (stop_condition<0.0001 and res['success']) or t+1>=maxiter1:#
			#if out1>out2[0]:
			p_B,p_I,W,r,infer_data=out2[1],out2[2],out2[3],out2[4],out2[5]	
			scale=mean(out2[5][:,:incub],axis=1)
			net_scaled=dot(diag(1./scale),W).dot(diag(scale))
			#else:
			
				
			break
			
		else:
			p_B,p_I,W,r,infer_data=p_B1,p_I1,net1,recovery1,dd1
			t+=1
	is_forecast=gen_forecast(infer_data,[W,r],incub,p_I,p_B)
	fig=plt.figure('observed infection forecast '+name)
	ax=fig.add_subplot(121)
	ax.set_title('observed infection fitting')
	ax.plot(arange(data.shape[1]),dot(ones(data.shape[0]-1),data[non_wuhan]),c='r',linestyle='-',marker='o',label='True Infection number')
	ax.plot(arange(data.shape[1]),dot(ones(data.shape[0]-1),is_forecast[non_wuhan]),c='b',linestyle='--',marker='^',label='Expected Infection number')
	ax.set_xticks(arange(data.shape[1]))
	ax.set_xticklabels(list(array([datetime.datetime.strftime(x,'%b-%d') for x in date_index])))
	for tick in ax.get_xticklabels():
		tick.set_rotation(90)
	ax.legend(fontsize=5,prop={'size':5})
	ax=fig.add_subplot(122)
	ax.set_title('hidden infection forecast')
	is_forecast=gen_forecast(infer_data,[W,r],incub,p_I,p_B,hidden=True)
	
	ax.plot(arange(is_forecast.shape[1]),dot(ones(is_forecast.shape[0]),is_forecast),c='b',linestyle='--',marker='^',label='Expected Infection number')
	os_forecast=OS_forecast(is_forecast,p_I,W,r,incub,os_T)
	ax.plot(arange(is_forecast.shape[1]-1,is_forecast.shape[1]+os_T),dot(ones(os_forecast.shape[0]),os_forecast),c='k',linestyle='--',marker='^',label='Expected Future Infection number')
	ax.set_xticks(arange(is_forecast.shape[1]+os_T))
	ax.set_xticklabels(['pre_'+str(i) for i in range(14)[::-1]]+list(array([datetime.datetime.strftime(x,'%b-%d') for x in date_index]))+['f_'+str(i) for i in range(20)])
	for tick in ax.get_xticklabels():
		tick.set_rotation(90)
	ax.legend(fontsize=5,prop={'size':5})
	
	return p_B,p_I,W,net_scaled,scale,r,is_forecast,os_forecast,infer_data
	
def OS_forecast(data,p_I,W,r,incub,os_T):
	# generate out-sample forecast (forecast length = os_T) for infection number 
	# given W, p_I and the hidden infection number (data) in the last incub days
	delta=data
	out=[]
	for i in range(os_T):
		o1=delta[:,-1]+dot(dot(W,delta[:,-incub:]),p_I)-dot(diag(r),delta[:,-1])
		delta=append(delta,o1.reshape(len(o1),1),axis=1)
		
	return delta[:,-os_T-1:]
	
	
##processing data
#process data
data=pd.ExcelFile(u'Wuhan_nCoV分省20200308.xlsx')
sheets=[u'湖北',u'直辖市',u'浙江江苏山东',u'广东',u'福建',u'新疆青藏港澳台海外',u'云南四川湖南广西贵州海南',u'安徽江西河北河南山西陕西',u'宁夏黑吉辽内蒙古甘肃']#[u'汇总表格',u'治愈病例']#[u'湖北',u'直辖市',u'浙江江苏山东',u'广东',u'福建',u'新疆青藏港澳台海外',u'云南四川湖南广西贵州海南',u'安徽江西河北河南山西陕西',u'宁夏黑吉辽内蒙古甘肃']
def map_date(x):
	for it in x.index:
		if type(x.loc[it,'date']) is float or x.loc[it,'date']=='':
			print x.loc[it,'date']
			x.loc[it,'date']=x.loc[it,u'新闻日期']
			if u'上午' in x.loc[it,'date']:
				x.loc[it,'date']=x.loc[it,'date'].replace(u'上午','')
			if u'下午' in x.loc[it,'date']:
				x.loc[it,'date']=x.loc[it,'date'].replace(u'下午','')
		else:
			x.loc[it]=array(x.loc[it])
	return x
def map_date1(x):
	for it in x.index:
		if type(x.loc[it,u'出院通报日期']) is float or x.loc[it,u'出院通报日期']=='':
			print x.loc[it,u'出院通报日期']
			x.loc[it,u'出院通报日期']=x.loc[it,u'出院日期']
			if u'上午' in x.loc[it,u'出院通报日期']:
				x.loc[it,u'出院通报日期']=x.loc[it,u'出院通报日期'].replace(u'上午','')
			if u'下午' in x.loc[it,u'出院通报日期']:
				x.loc[it,u'出院通报日期']=x.loc[it,u'出院通报日期'].replace(u'下午','')
	return x
def checkin(x,xlist):
	init=0
	if type(x) is float:
		return init
	else:
		for item in xlist:
			if x in item:
				init=1
				break
		return init
for i,it in enumerate(sheets):
	
	dd=data.parse(it)
		
	
	dd=dd[[u'地级行政单位',u'确诊日期',u'新闻日期',u'病例数',u'省份']]
	print dd[dd[u'省份']==u'上海']
	if it in [u'新疆青藏港澳台海外']:
		dd['in']=dd[u'省份'].map(lambda x:checkin(x,[u'新疆',u'青海',u'西藏',u'香港',u'澳门',u'台湾']))
		dd[u'省份'][dd['in']==0]='others'
		#dd=dd[dd['in']==1]
	dd.index=arange(len(dd))
	dd.rename(columns={u'省份':'city',u'确诊日期':'date',u'病例数':'number'},inplace=True)
	print dd[dd['city']==u'上海']	
	dd=dd[['city','number','date']]
	
	print dd.columns
	dd=dd.dropna(subset=['date'])
	if i==0:
		inp=dd
	else:
		inp=pd.concat([inp,dd])
inp['city']=inp['city'].map(lambda x:x.replace('\n','').replace(u'省',''))
		
cities=inp['city'].unique()
def handle_date(x):
	#try:
	#print x
	
	try:
		y=str(x)
	except Exception:
		if u'上午' in x:
			#print x
			x=x.replace(u'上午','')
		if u'下午' in x:
			#print x
			x=x.replace(u'下午','')
		#print x
		y=str(x)
	#except Exception as e:
		
	#	y=x
	#.split(' ')[0]
	
	#m=y.split('/')[1]
	#m='0'+m
	#out=y.split('/')
	#out[1]=m
	#return datetime.datetime.strptime(y,'%Y-%m-%d %H:%M:%S')
	#try:
	if '/' in y:
		if y[8]==' ':
			y=y[:8].replace(' ','/')
			#print y
		else:
			y=y[:9].replace(' ','/')
			#print y
		return datetime.datetime.strptime(y,'%Y/%m/%d')#'-'.join(out),'%Y-%m-%d')#y,'%Y-%m-%d')#
	else:
		y=y[:10]
		return datetime.datetime.strptime(y,'%Y-%m-%d')
	#except Exception as e:
	#	print e
	#	return nan
print inp['date']
inp['date']=inp['date'].map(lambda x:handle_date(x))
#inp=inp.dropna(subset=['date'])
ini=min(inp['date'])
end=max(inp['date'])#datetime.datetime.strptime('2020-02-12','%Y-%m-%d')#
print ini,end


inp['time']=inp['date'].map(lambda x:int((x-ini).days))

between=(end-ini).days
inp=inp.groupby(['city','date','time'],as_index=False).agg(sum)
def process_time_series(data,number):
	data1=[]
	data.index=data['time']
	for i in range(between):
		if i in data['time']:
			data1.append(array(data.loc[i,['city',number,'date','time']]))
		else:
			data1.append(array([array(data['city'])[0],0,ini+datetime.timedelta(days=i),i]))
	data1=array(data1)
	print data1.shape#,array(data)
	data1=pd.DataFrame(data1,columns=['city',number,'date','time'])
	return data1

inp=inp.groupby('city',as_index=False).apply(lambda x:process_time_series(x,'number'))	

inp.sort_values(['city','time'],inplace=True)
print inp[inp['city']==u'上海']

since_day=0
inp=inp[inp['time']>=since_day]
report_data=[]
city_index=inp['city'].unique()
print u'湖北' in city_index
wuhan=where(array(city_index)==u'湖北')[0][0]
non_wuhan=[i for i in range(len(city_index)) if i != wuhan]
date_index=inp['date'].unique()#.map(lambda x:datetime.datetime.strftime(x,'%b-%d')).unique()#.map(lambda x:str(x).split('-')[1].split(' ')[0]).unique()
lc,ld=len(city_index),len(date_index)
print date_index
print lc,ld

ini=min(inp['date'])
end=max(inp['date'])#datetime.datetime.strptime('2020-02-12','%Y-%m-%d')#
print ini,end

for i in city_index:
	d=array(inp[(inp['city']==i)]['number'])
	print d
	report_data.append(d)
	
report_datas=cumsum(array(report_data),axis=1)
report_data=report_datas.copy()
maxiter=13
maxiter1=13
os_T=1
incub=14
incub1=7
periods=[]
num=ld/os_T
forecast=True#False#False#
if forecast:
	forecasts=[]
	num=ld/os_T
	
	if since_day==0:
		writer=pd.ExcelWriter(result8.xlsx')
	else:
		writer=pd.ExcelWriter(result8_since_day'+str(since_day)+'.xlsx')
	for i in range(num):
		if os_T*(i)+incub1<=report_datas.shape[1]:
			report_data=report_datas[:,report_datas.shape[1]-incub1-os_T*i:report_datas.shape[1]-os_T*i]
			name=datetime.datetime.strftime(array(date_index)[max([report_datas.shape[1]-incub1-os_T*i,0])],'%b-%d')+'--'+datetime.datetime.strftime(array(date_index)[report_datas.shape[1]-os_T*i-1],'%b-%d')
			d_index=array(date_index)[report_datas.shape[1]-incub1-os_T*i:report_datas.shape[1]-os_T*i]
		elif incub1+os_T*(i-1)>report_datas.shape[1]:
			break
		else:
			report_data=report_datas[:,:report_datas.shape[1]-os_T*i]
		
			name=datetime.datetime.strftime(array(date_index)[0],'%b-%d')+'--'+datetime.datetime.strftime(array(date_index)[report_datas.shape[1]-os_T*i-1],'%b-%d')
			d_index=array(date_index)[:report_datas.shape[1]-os_T*i]		
		if i >0:
			for l in range(min([1,len(forecasts)])):#(2*incub-os_T)/os_T,len(forecasts)])):
				if report_data.shape[1]+incub<forecasts[-l-1].shape[1]-os_T*(l+1):
					starting=forecasts[-l-1].shape[1]-os_T*(l+1)-report_data.shape[1]-incub
				else:
					starting=0
				
				if l==0:
					
					pre_infer=forecasts[-l-1][:,starting:forecasts[-l-1].shape[1]-os_T*(l+1)]
				else:
					
					a=forecasts[-l-1][:,starting:forecasts[-l-1].shape[1]-os_T*(l+1)]
					
					pre_infer[:,-a.shape[1]:]=(pre_infer[:,-a.shape[1]:]*l+a)/float(l+1)
					
					
			
			
			print pre_infer.shape		
		else:
			last_net=None
			last_prob=None
			last_boom=None
			pre_infer=None
		boom_prob,infect_prob,net,net_scaled,scale,recovery,is_forecast,os_forecast,infer_data=snd_step_est(report_data,incub,maxiter,os_T,name,d_index,pre_infer=pre_infer,last_net=last_net)
		last_net=net
		last_prob=append(infect_prob,recovery[:1])
		last_boom=boom_prob
		forecasts.append(is_forecast)
		prob=pd.DataFrame(array([arange(incub),boom_prob,infect_prob]).T,columns=['delay','boom','infect'])
		net=pd.DataFrame(append(array(city_index).reshape(len(city_index),1),net,axis=1),columns=['city']+list(array(city_index)))
		recovery=pd.DataFrame(array([array(city_index),recovery]).T,columns=['city','recovery'])
		is_forecast=pd.DataFrame(append(array(city_index).reshape(len(city_index),1),append(is_forecast,os_forecast[:,1:],axis=1),axis=1),columns=['city']+['pre_'+str(j) for j in range(1,incub+1)[::-1]]+list(d_index)+['f_'+str(j) for j in range(os_T)])
		report_data=pd.DataFrame(append(array(city_index).reshape(len(city_index),1),report_data,axis=1),columns=['city']+list(d_index))
		infer_data=pd.DataFrame(append(array(city_index).reshape(len(city_index),1),infer_data,axis=1),columns=['city']+['pre_'+str(j) for j in range(1,incub+1)[::-1]]+list(d_index))
		net_scaled=pd.DataFrame(append(append(array(city_index).reshape(len(city_index),1),net_scaled,axis=1),scale.reshape(lc,1),axis=1),columns=['city']+list(array(city_index))+['scale'])
		
		prob.to_excel(writer,sheet_name='prob_'+str(os_T*i),index=False)
		net.to_excel(writer,sheet_name='ad_matrix_'+str(os_T*i),index=False)
		net_scaled.to_excel(writer,sheet_name='scaled_ad_matrix_'+str(os_T*i),index=False)
		recovery.to_excel(writer,sheet_name='recovery_'+str(os_T*i),index=False)
		is_forecast.to_excel(writer,sheet_name='forecast_'+str(os_T*i),index=False)
		report_data.to_excel(writer,sheet_name='raw_data_'+str(os_T*i),index=False)
		infer_data.to_excel(writer,sheet_name='hidden_data_'+str(os_T*i),index=False)
	writer.save()
	plt.show()
else:
	if since_day==0:
		writer=pd.ExcelFile(result8.xlsx')
	else:
		writer=pd.ExcelFile(result8_since_day'+str(since_day)+'.xlsx')
	for i in range(num):
		if os_T*(i)+incub1<=report_datas.shape[1]:
			report_data=report_datas[:,report_datas.shape[1]-incub1-os_T*i:report_datas.shape[1]-os_T*i]
			name=datetime.datetime.strftime(array(date_index)[max([report_datas.shape[1]-incub1-os_T*i,0])],'%b-%d')+'--'+datetime.datetime.strftime(array(date_index)[report_datas.shape[1]-os_T*i-1],'%b-%d')
			d_index=array(date_index)[report_datas.shape[1]-incub1-os_T*i:report_datas.shape[1]-os_T*i]
		elif incub1+os_T*(i-1)>report_datas.shape[1]:
			break
		else:
			report_data=report_datas[:,:report_datas.shape[1]-os_T*i]
		
			name=datetime.datetime.strftime(array(date_index)[0],'%b-%d')+'--'+datetime.datetime.strftime(array(date_index)[report_datas.shape[1]-os_T*i-1],'%b-%d')
			d_index=array(date_index)[:report_datas.shape[1]-os_T*i]		
		prob=writer.parse('prob_'+str(os_T*i))
		boom_prob=array(prob['boom'])
		infect_prob=array(prob['infect'])
		recovery=writer.parse('recovery_'+str(os_T*i))['recovery']
		forecast=writer.parse('forecast_'+str(os_T*i))
		is_forecast=array(forecast[[item for item in forecast.columns if 'f_' not in str(item) and str(item)!='city']],dtype=float)
		os_forecast=array(forecast[[item for item in forecast.columns if 'f_' in str(item) and str(item)!='city']])
		forecast=[]
		for j in range(report_data.shape[0]):
			
			o2=dot(moving_matrix(is_forecast[j],incub),boom_prob)
			forecast.append(o2)
		forecast=array(forecast)
		fig=plt.figure('observed infection forecast '+str(os_T*i))
		ax=fig.add_subplot(121)
		ax.set_title('observed infection fitting')
		#ax.plot(arange(report_data.shape[1]),dot(ones(report_data.shape[0]-1),report_data[non_wuhan]),c='r',linestyle='-',marker='o',label='True Infection number')
		#ax.plot(arange(report_data.shape[1]),dot(ones(report_data.shape[0]-1),forecast[non_wuhan]),c='b',linestyle='--',marker='^',label='Expected Infection number')
		ax.plot(arange(report_data.shape[1]),dot(ones(report_data.shape[0]),report_data),c='r',linestyle='-',marker='o',label='True Infection number')
		ax.plot(arange(report_data.shape[1]),dot(ones(report_data.shape[0]),forecast),c='b',linestyle='--',marker='^',label='Expected Infection number')
		ax.set_xticks(arange(report_data.shape[1]))
		ax.set_xticklabels(list(array([datetime.datetime.strftime(x,'%b-%d') for x in d_index])),fontsize=6)
		for tick in ax.get_xticklabels():
			tick.set_rotation(90)
		#ax.set_ylim((-1,13000))		
		ax.legend(loc='best',prop={'size': 7},fontsize=6)
		ax=fig.add_subplot(122)
		ax.set_title('hidden infection forecast')
		
		ax.plot(arange(is_forecast.shape[1]),dot(ones(is_forecast.shape[0]),is_forecast),c='b',linestyle='--',marker='^',label='Expected Infection number')
		ax.plot(arange(is_forecast.shape[1]-1,is_forecast.shape[1]+os_T),dot(ones(os_forecast.shape[0]),append(is_forecast[:,-1:],os_forecast,axis=1)),c='k',linestyle='--',marker='^',label='Expected Future Infection number')
		ax.set_xticks(arange(is_forecast.shape[1]+os_T))
		ax.set_xticklabels(['pre_'+str(j) for j in range(incub)[::-1]]+list(array([datetime.datetime.strftime(x,'%b-%d') for x in d_index]))+['f_'+str(j) for j in range(os_T)],fontsize=6)
		for tick in ax.get_xticklabels():
			tick.set_rotation(90)
		ax.legend(loc='best',prop={'size': 7},fontsize=6)
	plt.show()