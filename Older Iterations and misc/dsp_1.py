import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, butter,lfilter, freqz, find_peaks

import matplotlib.animation as animation






    
global timer
global dsp_array
x=[]
y=[]
   
    

i=0#int(len(dsp_array)/2)
    
    

fig=plt.figure(num='side-ways Drift Activity')
ax=fig.add_subplot(111)
fig2=plt.figure(num='Wave Activity')
ax2=fig2.add_subplot(111)
#fig3=plt.figure()
#ax3=fig3.add_subplot(111)


fig4=plt.figure(num='Swell Activity')
ax4=fig4.add_subplot(111)


order = 3
fs =30#10# 30   # sample rate, Hz
cutoff =1.2#1 #3


'''
    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, signalc)
'''



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



def swell_count(swell_filt,lpsw2,lp2):
    iter=0
    while iter<len(lp2):
        swell_filt.append((int(lpsw2[iter])-int(lp2[iter]))*-1)
        #remove values when swell is bigger than the break(noise)
        if swell_filt[iter]<0:
            swell_filt[iter]=swell_filt[iter]*-1
            
            
        iter+=1
    max_swell= max(swell_filt)
    iter=0
    while iter<len(lp2):
        swell_filt[iter]=swell_filt[iter]/max_swell
        
        
        if swell_filt[iter]<0.25 or swell_filt[iter]>1  :
            swell_filt[iter]=0
        iter+=1
    iter=0
    peaks, _ = find_peaks(swell_filt, height=None, threshold=None,\
    distance=None, prominence=None, width=1, wlen=None, rel_height=0.5, plateau_size=None)
    print('Number of Peaks: ', len(peaks))
       
    return swell_filt,peaks
        
def break_filter(break_filt,lpsw2,lp2):
    iter=0
    while iter<len(lp2):
        break_filt.append((int(lp2[iter])))#*int(lpsw2[iter])))#*-1)
        peaks, _ = find_peaks(break_filt, height=None, threshold=None,\
        distance=None, prominence=None, width=1, wlen=None, rel_height=0.5, plateau_size=None)
        iter+=1
    print('All-Peak Count: ', len(peaks))
            
            
        
    return break_filt
    
    


def Data(i):
    

    
    graph_data = open('DSP_data_AVG.txt','r').read()
    lines=graph_data.split('\n')
    y=[]
    sw=[]
    x=[]
    swell_filt=[]
    break_filt=[]
    drift_val=[]
    iter=0
    
   # if len(lines)>1800:
     #   while len(lines)>1800:
        
                
    for line in lines:

     # if len(lines)%30==0:
     #   fig.clear
        if len(line)>1:
            v,vs,drft,t=line.split(',')
            
            if iter% 15==0:#int(fsmap/30)
               # print(line)
                y.append(int(v))
                sw.append(int(vs))
                x.append(float(t))#x
                drift_val.append(int(drft))
                
            iter+=1
            
    AVGSw=sum(sw)/len(sw)
    
    maxSW=max(sw)
    AVGY=sum(y)/len(y)
    

  #  fig3.title('Raw Data',)
 #   ax3.plot(x,sw,color='b')   
 #   ax.plot(x, sw,color='g')        
  #  ax3.plot(x, y,color='y')  
      #  for iter<120: 
      #      iter+=1
      #  iter=0
    
    
    
    
   
    
    lp_drt=butter_lowpass_filter(drift_val, cutoff, fs, order) 
    lp2 = butter_lowpass_filter(y, cutoff, fs, order) 
    lpsw2 = butter_lowpass_filter(sw, cutoff, fs, order)
    swell_filt,peaks=swell_count(swell_filt,lpsw2,lp2)
    break_filt=break_filter(break_filt,lpsw2,lp2)
    ax4.plot(x,swell_filt,color='b')   
    fig4.canvas.draw()  
        
    ax.plot(lp_drt,x,color='g')    
    fig.canvas.draw()
    ax2.plot(x,break_filt,color='r')   
  #  ax2.set_xlim(left=max(0,(len(x)/30)-60),right=((len(x)/15)+5))
   # ax2.plot(x, lp2,color='b')   
 #   ax.plot(x, sw,color='g')        
   # ax2.plot(x, lpsw2,color='y')   
    fig2.canvas.draw()
    
    ax.set_ylim(bottom=max(0,(len(lp2)/2)-60),top=(len(lp2)/2)+5)
   # ax.set_ylim(left=max(0,(len(lp2)/2)-60),right=(len(lp2)/2)+5)
    ax2.set_xlim(left=max(0,(len(lp2)/2)-60),right=(len(lp2)/2)+5)
    ax4.set_xlim(left=max(0,(len(lp2)/2)-60),right=(len(lp2)/2)+5)
ani=animation.FuncAnimation(fig,Data,interval=10000)    
ani=animation.FuncAnimation(fig2,Data,interval=10000)
ani=animation.FuncAnimation(fig4,Data,interval=10000)
#fig.show()
        









 #   maxY=max(y)
#    wellit=0
#    while wellit <len(sw):
 #       sw[wellit]=sw[wellit]-AVGSw
 #       sw[wellit]=sw[wellit]/maxSW
 #       y[wellit]=y[wellit]-AVGY
 #       y[wellit]=y[wellit]/maxY
  #      wellit+=1
 #   minY=min(y)
#    minSW=min(sw)
 #   wellit=0  
#    while wellit <len(sw):
 #       sw[wellit]=sw[wellit]-minY
 #       y[wellit]=y[wellit]-minSW

 #       wellit+=1
 #   wellit=0 
#    lp = butter_lowpass_filter(y, cutoff, fs, order) 
#    lpsw = butter_lowpass_filter(sw, cutoff, fs, order) 
     
 
       #  lp = butter_lowpass_filter(y, cutoff, fs, order)
  #  ax.plot(x, y ,color='r')
#    ax.plot(x, lp,color='b')   
 #   ax.plot(x, sw,color='g')        
  #  ax.plot(x, lpsw,color='y')   
   # ax.set_xlim(left=max(0,i-1600),right=i+5)
       # fig.clear(n)    
  #  fig.canvas.draw()