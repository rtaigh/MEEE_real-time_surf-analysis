import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, butter,lfilter, freqz, find_peaks
from matplotlib import style
import matplotlib.animation as animation




style.use('dark_background')


    
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


order = 6
fs =6#30#10# 30   # sample rate, Hz
cutoff =0.25#1.2#1 #3
Swell_cutoff=0.8
minframe_setup=fs*60


            
    
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
    distance=6, prominence=None, width=1, wlen=None, rel_height=0.5, plateau_size=None)
   # print('Number of Peaks: ', len(peaks))
       
    return swell_filt,peaks



        
def break_filter(break_filt,lpsw2,lp2):
    iter=0
    while iter<len(lp2):
        break_filt.append((int(lp2[iter])))#*int(lpsw2[iter])))#*-1)
        peaks, _ = find_peaks(break_filt, height=None, threshold=None,\
        distance=None, prominence=None, width=2, wlen=None, rel_height=0.5, plateau_size=None)
        iter+=1
    #print('peaks: ', peaks)
  #  print('All-Peak Count: ', len(peaks))
            
            
        
    return break_filt,peaks
    
    


def Data(i):
    #try:
        
        ax.clear()
        ax2.clear()
        ax4.clear()
        
        graph_data = open('2019_3_30_Full_day_data.txt','r').read()#Minute_history.txt#2019_3_29_Full_day_data.txt
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
            
                    #5400
        for line in lines[len(lines)-5400:len(lines)]:#saves on memory
           # print(line)
         # if len(lines)%30==0:
         #   fig.clear
            if len(line)>1:
                v,vs,drft,t=line.split(',')
                if iter% 5==0:#int(fsmap/30)
                   # print(line)
                    y.append(int(v))
                    sw.append(int(vs))
                    x.append(float(t))#x#t
                    drift_val.append(int(drft))
                   # print(y,sw,x)
                iter+=0.5
       # AVGSw=sum(sw)/len(sw)
        
       # maxSW=max(sw)
       # AVGY=sum(y)/len(y)
  
        lp_drt=butter_lowpass_filter(drift_val, cutoff, fs, order) 
        lp2 = butter_lowpass_filter(y, Swell_cutoff, fs, order) 
        
        lpsw2 = butter_lowpass_filter(sw, cutoff, fs, order)
        swell_filt,swell_peaks=swell_count(swell_filt,lpsw2,lp2)
        break_filt,break_peaks=break_filter(break_filt,lpsw2,lp2)
        
        break_plot=break_filt[len(break_filt)-60:len(break_filt)]
        swell_plot=swell_filt[len(swell_filt)-60:len(swell_filt)]
        time_plot=x[len(swell_filt)-60:len(swell_filt)]
        ax4.plot(time_plot,swell_plot,color='b')
        ax4.set_title('Clean Swell Activity')
        ax4.set_xlabel('Time (Sec)')
        ax4.set_ylabel('Swell Presence')
        ax4.set_yticklabels([])
        
        
        Swell_plot_peaks, _ = find_peaks(swell_plot, height=None, threshold=None,\
        distance=4, prominence=None, width=2, wlen=None, rel_height=0.5, plateau_size=None)
          
        Break_plot_peaks, _ = find_peaks(break_plot, height=None, threshold=None,\
        distance=4, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None) 
        
        
        
        
            # ax2.set_xticklabels(['T+10','T+20','T+30','T+40','T+50','T+60','T+70','T+80','T+90','T+100'])
       # ax4.set_xticklabels(['T+10','T+20','T+30','T+40','T+50','T+60','T+70','T+80','T+90','T+100'])
        ax2.set_title('Break Activity')
        ax2.set_xlabel('Time (Sec)')
        ax2.set_ylabel('Break height and Amplitude')
        #ax2.set_adjustable(break_plot,share=False)
        ax2.plot(time_plot,break_plot,color='r') 
        
        
        
        
        for peak in Swell_plot_peaks:
            ax4.plot( time_plot[peak],swell_plot[peak], "o",color='r')
    
        for peak in Break_plot_peaks:
            ax2.plot(time_plot[peak],break_plot[peak], "o",color='g')       
    
         
        
       # fig.update()
       # ax.set_yticklabels(['T+10','T+20','T+30','T+40','T+50','T+60','T+70','T+80','T+90','T+100'])
        ax.set_title('Drift Activity') 
        ax.plot(lp_drt,x,color='y') 
        ax.set_ylabel('Time (Sec)')
        ax.set_xticklabels([])
        
       
        
   
        
        
        
        
        
        
        

    
        
   
        fig2.canvas.draw()
        fig4.canvas.draw() 
        fig.canvas.draw() 
        
     #   ax.set_ylim(bottom=max(0,(len(break_filt)-60),top=(len(break_filt))+5))
      #  ax2.set_xlim(left=max(0,(len(break_filt)-60),right=(len(break_filt))+5))
      #  ax4.set_xlim(left=max(0,(len(break_filt)-60),right=(len(break_filt))+5))
      #  ax4.set_xlim(left=max(0,(len(lp2)/2)-60),right=(len(lp2)/2)+5)
        
            
            
ani=animation.FuncAnimation(fig,Data,interval=1000)    
ani=animation.FuncAnimation(fig2,Data,interval=1000)
ani=animation.FuncAnimation(fig4,Data,interval=1000)
        
        




