# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:00:20 2023

@author: Publico
"""


import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import pyvisa as visa
import math
import time
from instrumental import TDS1002B
rm = visa.ResourceManager()
instrumentos = rm.list_resources()
print(instrumentos)
#osciloscopio
osci = TDS1002B('USB0::0x0699::0x0363::C065092::INSTR')

#%%
#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)
#%%
#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev5/ai1",max_val=5,min_val=-5)
    #ai_channel = task.ai_channels.add_ai_voltage_chan("Dev7/ai0",max_val=0.1,min_val=-0.1)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)	

## Medicion por tiempo/samples de una sola vez
def medicion_una_vez(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
        task.ai_channels.add_ai_voltage_chan("Dev5/ai1", terminal_config = modo, max_val=5,min_val=-5)
        #task.ai_channels.add_ai_voltage_chan("Dev7/ai0", terminal_config = modo, max_val=0.2,min_val=-0.2)
               
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
    datos = np.asarray(datos)    
    return datos

duracion = 1 #segundos
fs = 5000 #Frecuencia de muestreo
y = medicion_una_vez(duracion, fs)
plt.plot(y[0,:])
plt.plot(y[1,:])
plt.grid()
plt.show()

#%%
## Medicion continua
def medicion_continua(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev5/ai1", terminal_config = modo)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        while total<cant_puntos:
            time.sleep(0.1)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
            total = total + len(datos)
            t1 = time.time()
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))            

fs = 250000 #Frecuencia de muestreo
duracion = 10 #segundos
medicion_continua(duracion, fs)
#%%
plt.close("all")
osci.get_time()
#osci.set_time(scale = 1e-3)
#osci.set_channel(1,scale = 2)
tiempo, data = osci.read_data(channel = 1)
lctm= np.array([tiempo,data])
plt.plot(tiempo,data)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.ylim(osci.get_range(channel = 1))
np.savetxt('datosbismuto50usdiv.txt', np.transpose(lctm))
#%%
iteraciones=np.zeros(300)
for i in range(len(iteraciones)):
    p=1 + i
    osci.get_time()
    #osci.set_time(scale = 1e-3)
    #osci.set_channel(1,scale = 2)
    tiempo, data = osci.read_data(channel = 1)
    lctm= np.array([tiempo,data])
    #plt.figure()
    #plt.plot(tiempo,data)
    #plt.xlabel('Tiempo [s]')
    #plt.ylabel('Voltaje [V]')
    #plt.ylim(osci.get_range(channel = 1))
    np.savetxt(str(p)+'datosbismuto50usdiv.txt', np.transpose(lctm))
    time.sleep(0.001)