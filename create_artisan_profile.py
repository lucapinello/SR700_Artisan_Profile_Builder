#Luca Pinello 2019, @lucapinello
#GPLv3 license

from scipy import optimize, interpolate
from collections import OrderedDict
import numpy as np
import scipy as sp

import sys
import matplotlib
matplotlib.use('Qt4Agg')
import pylab as plt
from pylab import plot, ginput, show, axis

import json

from matplotlib.widgets import Cursor


flat_list = lambda l: [item for sublist in l for item in sublist]
use_log=False

def write_artisan_alarm_profile(alarm_filename,seconds,temp_profile,fan_profile,decimate=1,preheat_time=29):
    
    profile=OrderedDict()
    if decimate>1:
        seconds=seconds[::decimate]
        temp_profile=temp_profile[::decimate]
        fan_profile=fan_profile[::decimate]


    len_profile=len(temp_profile)*2
    

    
    #alarmactions
    #16 charge
    #6 temp
    #4 fan
    #13 drop
    #14 cool end

    profile['alarmactions']=[16]+[6,4]*len(temp_profile)+[13,14]


    #alarmstrings contains the temperature and fan values, 
    #here we assume that temp and fan are alternating
    profile['alarmstrings']=list(map(str,['charge']+flat_list(list(zip(temp_profile,fan_profile)))+['drop','cool end']))


    #alarmconds always 1
    profile['alarmconds']=[1]*(len_profile+3)

    #alarmnegguards and alarmguards  and alarmtimes  -1
    profile['alarmnegguards']=[-1]*(len_profile+3)
    profile['alarmguards']=[-1]*(len_profile+3)
    profile['alarmtimes']=[-1]*(len_profile+3)

    #alarmflags and alarmsources 1
    profile['alarmflags']=[1]*(len_profile+3)
    profile['alarmsources']=[1]*(len_profile+3)

    #alarmtemperatures always 500
    profile['alarmtemperatures']=[500.0]*(len_profile+3)

    #alarmoffsets is the time, we need to add 30 for accounting the preheat phase
    last_time=max(seconds)
    profile['alarmoffsets']=list(map(int,[preheat_time]+flat_list([ (a,a) for a in seconds])+[last_time+1,last_time+1+60*3]))

    #alarmbeep always 0
    profile['alarmbeep']=[0]*(len_profile+3)
    
    json.dump(profile,open(alarm_filename,'w+'))
    


try:
	n_points=int(sys.argv[1])
except:
	n_points=5
	

try:
	profile_filename=sys.argv[2]
except:
	profile_filename='artisan_profile_alarms.alrm'


min_time=30
max_time=750

x_min=min_time-30
x_max=max_time+30
y_min=100
y_max=450

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(1, 1, 1)
plt.xticks(np.arange(x_min,x_max,30))

plt.xlabel('Seconds')
plt.ylabel('Temperature in F°')

plt.title('Press spacebar to add sequentially %d points in the dotted area\nright click to undo' % n_points)


axis([x_min, x_max, y_min, y_max])
plt.plot([30,30],[y_min,y_max],'--k')
plt.plot([750,750],[y_min,y_max],'--k')
plt.plot([x_min,x_max],[150,150],'--k')
plt.plot([x_min,x_max],[400,400],'--k')
plt.grid(True)

cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

pts = ginput(n_points,show_clicks=True, mouse_pop=True) # it will wait for three clicks
plt.grid(True)
print ("The point selected are")
print(pts) # ginput returns points as tuples
x=list(map(lambda x: x[0],pts)) # map applies the function passed as 
y=list(map(lambda x: x[1],pts)) # first parameter to each element of pts
plt.close()

#sort points

idxs_sort=np.argsort(x)

x=list(np.array(x)[idxs_sort])
y=list(np.array(y)[idxs_sort])

xi=np.arange(30,750,1)

if use_log:
	[a,b]=optimize.curve_fit(lambda t,a,b: a+b*np.log10(t),  x,  y)[0]
	f_fit = lambda x: a + b*np.log10(x)
	y_new=f_fit(xi)
else:
	tck = interpolate.splrep(x, y, s=0)
	y_new = interpolate.splev(xi, tck, der=0)

fig = plt.figure(figsize=(11, 7))
plt.title('Close this plot to save the profile in: %s' % profile_filename)
ax = fig.add_subplot(1, 1, 1)
plt.xticks(np.arange(x_min,x_max,30))

plt.xlabel('Seconds')
plt.ylabel('Temperature in F°')

axis([x_min, x_max, y_min, y_max])
plt.plot([30,30],[y_min,y_max],'--k')
plt.plot([375,375],[y_min,y_max],'--k')
plt.plot([750,750],[y_min,y_max],'--k')
plt.plot([x_min,x_max],[150,150],'--k')
plt.plot([x_min,x_max],[400,400],'--k')
plt.grid(True)
plot(xi,y_new,'-')
plt.savefig('target_curve_for_%s.pdf' % profile_filename)
plt.show()
seconds=list(xi)
temp_profile=[max(150,a) for a in map(int,y_new)]
fan_profile=[min(9,a) for a in np.linspace(15,3,len(seconds)).astype(int)]

write_artisan_alarm_profile(profile_filename,seconds,temp_profile,fan_profile,decimate=20)

print('Profile saved!')
print('Please send comments or bug @lucapinello')
