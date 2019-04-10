import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter
from scipy.signal import find_peaks
from matplotlib import style
import matplotlib.animation as animation
import matplotlib.lines as mlines
from datetime import datetime




style.use('dark_background')


global timer
global dsp_array
x=[]
y=[]
   
    

i=0#int(len(dsp_array)/2)
    
    
#initialising plots
fig=plt.figure(num='side-ways Drift Activity')
ax=fig.add_subplot(111)
fig2=plt.figure(num='Wave Activity')
ax3=fig2.add_subplot(312)
ax2=fig2.add_subplot(311,sharex=ax3)
#fig2.


ax4=fig2.add_subplot(313)

#fig3=plt.figure()
#ax3=fig3.add_subplot(111)


#fig4=plt.figure(num='Swell Activity')
#ax4=fig4.add_subplot(111)


order = 5
fs =6#30#10# 30   # sample rate, Hz
cutoff =0.25#1.2#1 #3
Swell_cutoff=.25
#cutoff_so=0.25
plot_frame_size=fs*60#*60#convert to a minute value



    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


#Two methods for differentiating swell from breaks
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

    peaks, _ = find_peaks(swell_filt, height=None, threshold=None,distance=6, prominence=None, width=1, wlen=None, rel_height=0.5)#, plateau_size=None)
   # print('Number of Peaks: ', len(peaks))
       
    return swell_filt,peaks

def swell_count_2(swell_filt2,y,sw):
    iter=0
    while iter<len(sw):
        if ((int(sw[iter])-int(y[iter]))*-1)<0:
            swell_filt2.append(0)
        else: 
            swell_filt2.append(((int(sw[iter])-int(y[iter]))*-1))

        iter+=1    
    return swell_filt2
        
 
    


def Data(i):
    
       #Refresh Plots 
        ax.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        #Open and format File
        dt = datetime.now()
        Daily_data=(str(dt.year) +'_'+str(dt.month)+'_'+str(dt.day) +'_Full_day_data.txt')
        graph_data = open(Daily_data,'r').read()#Minute_history.txt#2019_3_29_Full_day_data.txt
        lines=graph_data.split('\n')
        y=[]
        sw=[]
        x=[]
        swell_filt=[]
        swell_filt2=[]
      #  break_filt=[]
        drift_val=[]
        iter=0
        
        
        #extract A sample of the data over the past minute of operations
        for line in lines[len(lines)-(plot_frame_size*30):len(lines)]:#saves on memory
            
     
            if len(line)>1:
                v,vs,drft,t=line.split(',')
                
                #Downsampling
              
                   # print(line)
                y.append(int(v))
                sw.append(int(vs))
                x.append(float(t))#x#t
                drift_val.append(int(drft))

                iter+=0.5

        
        #Applying low-pass filters to remove noise from incoming signal

        lp_drt=butter_lowpass_filter(drift_val, cutoff, fs, order) 
        lp2 = butter_lowpass_filter(y, Swell_cutoff, fs, order) 
        
        lpsw2 = butter_lowpass_filter(sw, cutoff, fs, order)
        swell_filt,swell_peaks=swell_count(swell_filt,lpsw2,lp2)
        
        # Sumtract to main signals
        swell_filt2=swell_count_2(swell_filt2,y,sw)
        #Apply low pass filter on the resulting output
        swell_filt2=butter_lowpass_filter(swell_filt2, cutoff, fs, order)
       # swell_filt2=butter_lowpass_filter(swell_filt2, cutoff, fs, order)

        #formatting plot to 60 seconds
        break_plot=lp2[(len(lp2)-(int(plot_frame_size))):len(lp2)]
        
        lp_drt=lp_drt[(len(lp_drt)-(int(plot_frame_size))):len(lp_drt)]
        time_plot=x[(len(x)-(int(plot_frame_size))):len(x)]
        
        #print(plot_frame_size)
        



        swell_only_plot=swell_filt2[(len(swell_filt2)-(int(plot_frame_size))):len(swell_filt2)]
       # swell_only_plot= butter_lowpass_filter(swell_only_plot, cutoff_so, fs, order)
        swell_plot=lpsw2[(len(lpsw2)-(int(plot_frame_size))):len(lpsw2)]
        
        
       
        
        Swell_plot_peaks, _ = find_peaks(swell_plot, height=None, threshold=None,\
        distance=4, prominence=5, width=None, wlen=None, rel_height=0.5)#, plateau_size=None)
          
        Break_plot_peaks, _ = find_peaks(break_plot, height=None, threshold=None,\
        distance=4, prominence=5, width=None, wlen=None, rel_height=0.5)#, plateau_size=None) 
        
        Swell_only_peaks, _ = find_peaks(swell_only_plot, height=8, threshold=0,\
        distance=10, prominence=8, width=5, wlen=None, rel_height=0.5)#, plateau_size=None)  
        
        #Formatting axis on display
        ax.set_yticklabels(['T+80','T+70','T+60','T+50','T+40','T+30','T+20','T+10'])
        ax.set_xticklabels(['Far Left','Left','','Centred','','Right','Far Right',''])
        ax3.set_xticklabels(['T+80','T+70','T+60','T+50','T+40','T+30','T+20','T+10'])
        
        ax2.set_yticklabels(['','','','',''])
        ax3.set_yticklabels(['','False','','True','','','','','','','',''])
        
        
        #ax4.set_xticklabels(['T+10','T+20','T+30','T+40','T+50','T+60','T+70','T+80','T+90','T+100'])
        ax2.set_title('Wave Activity') 
        ax2.set_ylabel('Break height and Amplitude')
        #ax2.set_adjustable(break_plot,share=False)
        ax2.plot(time_plot,lp2[(len(x)-(int(plot_frame_size))):len(x)],color='r')
        # ax2.plot(time_plot,break_plot,color='r') 
        ax2.plot(time_plot,lpsw2[(len(x)-(int(plot_frame_size))):len(x)],color='gold')
       # ax2.set_xticklabels([])
       # ax4.plot()

        
        
    
        for peak in Break_plot_peaks:
            ax2.plot(time_plot[peak],break_plot[peak], "o",color='r')       
       
        for peak in Swell_only_peaks:
            ax3.plot(time_plot[peak],swell_only_plot[peak], "s",color='g') 
       
        for peak in Swell_plot_peaks:
            ax2.plot( time_plot[peak],swell_plot[peak], "^",color='gold')
            
            
       #period calculations     
        try:
           Full_period=int(60/len(Break_plot_peaks))
           #Swell_period= int(60/len(Swell_only_peaks))
       
           Wave_height_max=break_plot[max(Break_plot_peaks)]
           Wave_height_min=break_plot[min(Break_plot_peaks)]
           Good_Wave_ratio=((len(Swell_only_peaks)/len(Swell_plot_peaks))*100)
        except:
           1+1
        if Good_Wave_ratio>100:
            Good_Wave_ratio=100
        
        if Wave_height_max<30:
            max_h_status='Not great at all'
        if Wave_height_max>30:
            max_h_status='Great'
        if Wave_height_max>60:
            max_h_status='Fantastic' 
        
        if Wave_height_min<30:
            min_h_status='Not great'
        if Wave_height_min>30:
            min_h_status='Okay'
        if Wave_height_min>60:
            min_h_status='Good'  
        
            
                
        if Full_period<15:
            wave_status='Incredible conditions'
        if Full_period<15:
            wave_status='Great conditions'
        if Full_period<12:
            wave_status='Good conditions'
        if Full_period<9:
            wave_status='Okay conditions'
        if Full_period<6:
            wave_status='Not Good'
        if Full_period<4:
            wave_status='No surf'
            
            
        ax3.plot(time_plot,swell_filt2[(len(x)-(int(plot_frame_size))):len(x)],color='g') 
        ax3.set_ylabel('Swell activity')
        ax3.set_xlabel('Time of event plus delay (Sec)')    
            
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)   
        ax4.axis('off')
        ax4.patch.set_visible(False) 
        try:
            ax4.text(0, 0,('\n\n Overall status: '+wave_status+'\n\n'+str(int(Good_Wave_ratio))+'% of the detected break is clean \n\n\
            A period of '+ str(Full_period) +' seconds between waves was detected,\n\n\
            Wave heights detected ranges from '+min_h_status+' to '+max_h_status), ha='left',fontsize=15, color='w', rotation=0, wrap=True)    
        except:
           ax4.text(0, 0,'Calibrating...', ha='left', color='white', rotation=0, wrap=True)
            
            
        gold_line = mlines.Line2D([], [], color='gold', marker='^',\
                          markersize=5, label='Swell Data with peaks')
        red_line = mlines.Line2D([], [], color='red', marker='o',\
                          markersize=5, label='Break Data with peaks')
        
        
        green_line = mlines.Line2D([], [], color='g', marker='s',\
                          markersize=5, label='Clean Swell Activity ')
        Ledg_content=fig2.legend(handles=[red_line,gold_line,green_line])
        fig2.gca().add_artist(Ledg_content)
        
        #fig2.text(.5,.05,wave_status,ha='center', va='bottom')
        

        
       # fig.update()
       # ax.set_yticklabels(['T+10','T+20','T+30','T+40','T+50','T+60','T+70','T+80','T+90','T+100'])
        ax.set_title('Drift Activity') 
        ax.plot(lp_drt,time_plot,color='b') 
        ax.set_ylabel('Time (Sec)')
        ax.set_xticklabels([])
    
        
        
        fig2.set_size_inches((6,10))
        fig.set_size_inches((7,4))
        fig2.canvas.draw()
        fig.canvas.draw() 
        

        
            
            
ani=animation.FuncAnimation(fig,Data,interval=3000)    
ani=animation.FuncAnimation(fig2,Data,interval=3000)

        
        




