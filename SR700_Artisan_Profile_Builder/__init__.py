#Luca Pinello 2019, @lucapinello
#GPLv3 license

from scipy import optimize, interpolate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from collections import OrderedDict
import numpy as np
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import pylab as plt
from pylab import plot, ginput, show, axis

import json

from matplotlib.widgets import Cursor


flat_list = lambda l: [item for sublist in l for item in sublist]

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

    profile['alarmactions']=[16]+[6,4]*len(temp_profile)+[13,14,15]


    #alarmstrings contains the temperature and fan values,
    #here we assume that temp and fan are alternating
    profile['alarmstrings']=list(map(str,['charge']+flat_list(list(zip(temp_profile,fan_profile)))+['drop','cool end','off']))


    #alarmconds always 1
    profile['alarmconds']=[1]*(len_profile+4)

    #alarmnegguards and alarmguards  and alarmtimes  -1
    profile['alarmnegguards']=[-1]*(len_profile+4)
    profile['alarmguards']=[-1]*(len_profile+4)
    profile['alarmtimes']=[-1]*(len_profile+4)

    #alarmflags and alarmsources 1
    profile['alarmflags']=[1]*(len_profile+4)
    profile['alarmsources']=[1]*(len_profile+4)

    #alarmtemperatures always 500
    profile['alarmtemperatures']=[500.0]*(len_profile+4)

    #alarmoffsets is the time, we need to add 30 for accounting the preheat phase
    last_time=max(seconds)
    profile['alarmoffsets']=list(map(int,[preheat_time]+flat_list([ (a,a) for a in seconds])+[last_time+1,last_time+1+60*3,last_time+5+60*3]))

    #alarmbeep always 0
    profile['alarmbeep']=[0]*(len_profile+4)

    json.dump(profile,open(alarm_filename,'w+'))


def main():

    print(\
'''
      _____ _____  ______ ___   ___
     / ____|  __ \|____  / _ \ / _ \\
    | (___ | |__) |   / / | | | | | |
     \___ \|  _  /   / /| | | | | | |
     ____) | | \ \  / / | |_| | |_| |
    |_____/|_|  \_\/_/   \___/ \___/
     /\        | | (_)
    /  \   _ __| |_ _ ___  __ _ _ __
   / /\ \ | '__| __| / __|/ _` | '_ \\
  / ____ \| |  | |_| \__ \ (_| | | | |
 /_/  __\_\_|   \__|_|___/\__,_|_| |_|
     |  __ \          / _(_) |
     | |__) | __ ___ | |_ _| | ___
     |  ___/ '__/ _ \|  _| | |/ _ \\
     | |   | | | (_) | | | | |  __/
    _|_|   |_|  \___/|_|_|_|_|\___|
   |  _ \      (_) |   | |
   | |_) |_   _ _| | __| | ___ _ __
   |  _ <| | | | | |/ _` |/ _ \ '__|
   | |_) | |_| | | | (_| |  __/ |
   |____/ \__,_|_|_|\__,_|\___|_|
''')

    print('SR700 Artisan Profile Builder - Luca Pinello 2019 (@lucapinello)\n\n')
    print('Send bugs, suggestions or *green coffee* to lucapinello AT gmail DOT com\n\n')

    parser = argparse.ArgumentParser(description='Parameters',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_points_temp',  type=int,  default=5)
    parser.add_argument('--n_points_fan',  type=int,  default=5)
    parser.add_argument('-n','--profile_filename',  help='Output name', default='artisan_profile_alarms.alrm')
    parser.add_argument('--start_time',  type=int,  default=30)
    parser.add_argument('--end_time',  type=int,  default=810)
    parser.add_argument('--min_temp',  type=int,  default=150)
    parser.add_argument('--max_temp',  type=int,  default=500)


    args = parser.parse_args()

    n_points_temp=args.n_points_temp
    n_points_fan=args.n_points_fan

    if n_points_temp==2:
        use_log=True
    else:
        use_log=False


    min_time=args.start_time
    max_time=args.end_time

    x_min=min_time-30
    x_max=max_time+30
    y_min_temp=args.min_temp-50
    y_max_temp=args.max_temp+50
    y_min_fan=0
    y_max_fan=10
    y_min_ror=0
    y_max_ror=50


    #plot to get points for temp
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(x_min,x_max,30),np.arange(x_min,x_max,30)/60,rotation = 45, ha="right")

    plt.xlabel('Minutes')
    plt.ylabel('Temperature in F°')

    plt.title('\nArtisan Profile Builder - Luca Pinello 2019\n\nPress the spacebar to add sequentially %d points inside the dotted area\nUse mouse right click to undo last point added' % n_points_temp)

    axis([x_min, x_max, y_min_temp, y_max_temp])
    plt.plot([args.start_time,args.start_time],[y_min_temp,y_max_temp],'--k')
    plt.plot([args.end_time,args.end_time],[y_min_temp,y_max_temp],'--k')
    plt.plot([x_min,x_max],[args.min_temp,args.min_temp],'--k')
    plt.plot([x_min,x_max],[args.max_temp,args.max_temp],'--k')
    plt.grid(True)

    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

    pts = ginput(n_points_temp,show_clicks=True, mouse_pop=True,timeout=0) # it will wait for three clicks
    plt.grid(True)
    print ("The points selected are:\n")
    print(pts) # ginput returns points as tuples
    x_temp=list(map(lambda x: x[0],pts)) # map applies the function passed as
    y_temp=list(map(lambda x: x[1],pts)) # first parameter to each element of pts
    plt.close()

    #sort points

    idxs_sort=np.argsort(x_temp)

    x_temp=list(np.array(x_temp)[idxs_sort])
    y_temp=list(np.array(y_temp)[idxs_sort])
    xi_seconds=np.arange(x_min,x_max,1)
    seconds=list(np.arange(min_time,max_time,1))

    if use_log:
    	[a,b]=optimize.curve_fit(lambda t,a,b: a+b*np.log10(t),  x_temp,  y_temp)[0]
    	f_fit = lambda x: a + b*np.log10(x)
    	temp_profile=f_fit(seconds)
    else:
    	tck_temp = interpolate.splrep(x_temp, y_temp, s=0)
    	temp_profile= interpolate.splev(seconds, tck_temp, der=0)

    #plot to get the obtained temp curve
    fig = plt.figure(figsize=(11, 7))
    pdf_filename='target_curve_for_%s.pdf' % args.profile_filename
    plt.title('\nArtisan Profile Builder - Luca Pinello 2019\n\nTemperature Curve')
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(x_min,x_max,30),np.arange(x_min,x_max,30)/60,rotation = 45, ha="right")

    plt.xlabel('Minutes')
    plt.ylabel('Temperature in F°')


    axis([x_min, x_max, y_min_temp, y_max_temp])
    plt.plot([args.start_time,args.start_time],[y_min_temp,y_max_temp],'--k')
    plt.plot([args.end_time,args.end_time],[y_min_temp,y_max_temp],'--k')
    plt.plot([x_min,x_max],[args.min_temp,args.min_temp],'--k')
    plt.plot([x_min,x_max],[args.max_temp,args.max_temp],'--k')
    plt.grid(True)

    temp_profile=[min(args.max_temp,max(args.min_temp,a)) for a in map(int,temp_profile)]
    plot(seconds,temp_profile,'-')
    plt.show()


    #plot to get points for fan
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(x_min,x_max,30),np.arange(x_min,x_max,30)/60,rotation = 45, ha="right")

    plt.xlabel('Minutes')
    plt.ylabel('Fan Speed (1-9)')

    plt.title('\nArtisan Profile Builder - Luca Pinello 2019\n\nPress the spacebar to add sequentially %d points inside the dotted area\nUse mouse right click to undo last point added' % n_points_fan)

    axis([x_min, x_max, y_min_fan, y_max_fan])
    plt.plot([args.start_time,args.start_time],[y_min_fan,y_max_fan],'--k')
    plt.plot([args.end_time,args.end_time],[y_min_fan,y_max_fan],'--k')
    plt.plot([x_min,x_max],[1,1],'--k')
    plt.plot([x_min,x_max],[9,9],'--k')
    plt.grid(True)

    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)

    pts = ginput(n_points_fan,show_clicks=True, mouse_pop=True,timeout=0) # it will wait for three clicks
    plt.grid(True)
    print ("The points selected are:\n")
    print(pts) # ginput returns points as tuples
    x_fan=list(map(lambda x: x[0],pts)) # map applies the function passed as
    y_fan=list(map(lambda x: x[1],pts)) # first parameter to each element of pts
    plt.close()

    #sort points

    idxs_sort=np.argsort(x_fan)

    x_fan=list(np.array(x_fan)[idxs_sort])
    y_fan=list(np.array(y_fan)[idxs_sort])

    f=interp1d(x_fan, y_fan, kind='previous', fill_value="extrapolate")
    fan_profile = f(seconds)
    fan_profile =list(map(int,[max(1,min(9,a)) for a in fan_profile ]))

    #plot to get the obtained fan curve
    fig = plt.figure(figsize=(11, 7))
    plt.title('\nArtisan Profile Builder - Luca Pinello 2019\n\nFan Curve')

    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(x_min,x_max,30),np.arange(x_min,x_max,30)/60,rotation = 45, ha="right")

    plt.xlabel('Seconds')
    plt.ylabel('Fan Speed (1-9)')

    axis([x_min, x_max, y_min_fan, y_max_fan])
    plt.plot([args.start_time,args.start_time],[y_min_fan,y_max_fan],'--k')
    plt.plot([args.end_time,args.end_time],[y_min_fan,y_max_fan],'--k')
    plt.plot([x_min,x_max],[1,1],'--k')
    plt.plot([x_min,x_max],[9,9],'--k')
    plt.grid(True)

    plot(seconds,fan_profile,'-')
    plt.show()
    plt.close()


    #plot with everything
    fig = plt.figure(figsize=(8*1.2, 10*1.2),dpi=72)

    #temp
    ax1 = fig.add_subplot(3, 1, 1)
    plt.title('\nArtisan Profile Builder - Luca Pinello 2019\n\nClose this plot to save the profile in: %s\nA pdf of these curves will be also saved in: %s\n\n Temperature Curve' % (args.profile_filename,pdf_filename))

    axis([x_min, x_max, y_min_temp, y_max_temp])
    ax1.xaxis.set_ticklabels([])
    plt.ylabel('Temperature in F°')
    axis([x_min, x_max, y_min_temp, y_max_temp])
    plt.xticks(np.arange(x_min,x_max,30),[])

    plt.yticks(np.arange(y_min_fan,y_max_temp,50))
    plt.grid(True)
    plot(seconds,temp_profile,'-r',lw=3)

    #fan
    ax2 = fig.add_subplot(3, 1, 2)
    plt.ylabel('Fan Speed (1-9)')
    plt.xticks(np.arange(x_min,x_max,30),[])


    axis([x_min, x_max, y_min_fan, y_max_fan])
    plt.yticks(np.arange(0,10,1))
    plt.title('Fan Curve')
    plt.grid(True)

    plot(seconds,fan_profile,'-g',lw=3)

    ax3 = fig.add_subplot(3, 1, 3)

    plt.xlabel('Minutes')
    plt.xticks(np.arange(x_min,x_max,30),np.arange(x_min,x_max,30)/60,rotation = 45, ha="right")
    plt.ylabel('Delta(BT)')
    plt.title('ROR Curve')
    plt.grid(True)


    ror=[ (temp_profile[i+30]-temp_profile[i]) for i in range(len(temp_profile)-30)]
    seconds_ror=seconds[:len(ror)]

    ror = savgol_filter(ror, 31, 3)
    plot(seconds_ror,ror,'-b',lw=3)

    plt.tight_layout()

    pdf_filename='target_curve_for_%s.pdf' % args.profile_filename
    plt.savefig(pdf_filename)

    plt.show()

    write_artisan_alarm_profile(args.profile_filename,seconds,temp_profile,fan_profile,decimate=20)

    print('\nProfile was saved in: %s.\n\nA pdf of this curve was saved in: %s!' % (args.profile_filename, pdf_filename))
    print('\nSend bugs, suggestions or *green coffee* to lucapinello AT gmail DOT com\n')
    print('Bye!\n')


if __name__ == "__main__":
    main()
