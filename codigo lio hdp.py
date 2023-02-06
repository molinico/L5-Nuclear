# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:34:31 2023

@author: lione
"""
#ESTE CODIGO ES EL MODIFICADO POR MI
#%%
import matplotlib.pyplot as plt
import numpy as np
#import nidaqmx
import math
import time
import scipy.signal
from scipy.optimize import curve_fit
import pandas as pd
import os
from statsmodels.stats.weightstats import DescrStatsW
#%%

#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices: 
    print(device)

with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev5/ai1",max_val=10,min_val=-10)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)
    

#%%

plt.close('all')

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev5/ai1",max_val=10,min_val=-10)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)	
	

## Medicion por tiempo/samples de una sola vez
#Nos pasaba que esta forma de sacar los datos andaba joya la primera vez y despues se rompia sin explicación:) capaz les ande igual
def medir(duracion, fs):
    cant_puntos = duracion*fs    
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
        task.ai_channels.add_ai_voltage_chan("Dev5/ai1", terminal_config = modo)
               
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE) #el modo FINITE te deja sacar la cantidad de puntos 
                                        #que quieras
        
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
    datos = np.asarray(datos)    
    return datos

duracion = 10 #segundos
fs = 250000 #Frecuencia de muestreo

y = medir(duracion, fs) #esto son los datos
xmax,ymax = scipy.signal.find_peaks(y,height = 0.01)

plt.scatter(xmax,y[xmax], color='red')
plt.plot(y)
plt.grid()
plt.show()

plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(0,1,50)#ACA ESTABLECES EL HILADO DEL HISTOGRAMA
axs[0].hist(y[xmax],bins = bins)
axs[0].set_title("Histograma picos")
axs[0].set_yscale("log")
axs[0].set_xlabel("Voltaje (v)")
axs[0].set_ylabel("Cuentas")
#axs[0].axis([1, 11, np.min(medicion_actual),np.max(medicion_actual)])


#%%
#TOMA CONTINUA DE DATOS
os.chdir(r"C:\Users\lione\OneDrive\Escritorio\lABO 5\nUCLEAR")
i=0
datosx = np.array([])
datosy= np.array([])

while i<=15:
    i=i+1
    duracion = 10 #segundos
    fs = 250000 #Frecuencia de muestreo
    y = medir(duracion, fs)
    xmax,ymax = scipy.signal.find_peaks(y,height = 0.01)
    datosx = np.concatenate([datosx, xmax], axis = None)
    datosy= np.concatenate([datosy, y[xmax]], axis = None)
    time.sleep(10)

np.savetxt('datobismuto4largo.txt', datosy) #ESTO TE CREA LO QUE QUERES

plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(0,1.2,50)
axs.hist(datosy,bins = bins)
axs.set_title("Histograma picos")
axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")
#axs[0].axis([1, 11, np.min(medicion_actual),np.max(medicion_actual)])

#%%
#si queres combinar varias tomadas de datos
plt.close('all')
cantidaddetxt= 4

j=1
datoscomb =np.array([])
while j < cantidaddetxt:
    datos = np.loadtxt('datoscesio' + str(j+1) +'.txt')
    print(len(datos))
    datoscomb = np.concatenate([datoscomb, datos], axis = None)
    j=j+1
    
print(len(datoscomb))

plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(0,1.2,75)
axs.hist(datoscomb,bins = bins)
axs.set_title("Histograma picos")
axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")

hist = axs.hist(datoscomb,bins=bins)
histx= np.array(hist[0])
histx =  np.insert(histx,1,0,axis=None)
histy= np.array(hist[1])
plt.figure()
plt.plot(histy,histx)
plt.yscale("log")
histcomp= [histx,histy]
xmax,ymax = scipy.signal.find_peaks(histx,height = 10000) #se va a cagar los xmax no sirven para nada, modificar la tol aca

#hago que xmax corresponda a los graficos
xmax_bien= []
ymax_bien= []
for i in range(len(xmax)):
    i=xmax[i]
    p=histy[i]
    q=histx[i]
    xmax_bien.append(p)
    ymax_bien.append(q)
    

plt.scatter(xmax_bien,ymax_bien)

#VEAMOS cuaL ES EL ancho cualitativamente
busqueda= 2

fotopico= xmax_bien[1] #FIJARSE ACA QUE SEA EL FOTOPCIO
err=[]
j=0
while j < len(histy):
    if histy[j] == fotopico:
        j_prima= j-busqueda
        j_primaf=j+busqueda
        zonax=histy[j_prima]
        zonaxf= histy[j_primaf]
        plt.scatter(zonax,histx[j_prima],c='#2ca02c')
        plt.scatter(zonaxf,histx[j_primaf],c='#2ca02c')
        j=j+1
        
        franja_x=[]
        franja_y=[]
        while j_prima<=j_primaf:
            franja_x.append(histy[j_prima])
            franja_y.append(histx[j_prima])
            j_prima=j_prima +1
        plt.figure()
        plt.plot(franja_x,franja_y, c='#d62728')
        err.append(franja_x[busqueda]-franja_x[0])
        #ploteemos el histo del fotopcio que no anda shit
        #plt.figure()
        #fig, axs = plt.subplots()
        #bins = np.linspace(franja_x[0],franja_x[len(franja_x)-1],len(franja_x))
        #axs.hist(franja_y,bins = bins)
        #axs.set_title("Histograma picos")
        #axs.set_yscale("log")
        #axs.set_xlabel("Voltaje (v)")
       # axs.set_ylabel("Cuentas")

    else:
        j=j+1

y_mean=np.mean(franja_y)
x_val=np.linspace(0.25,0.4,5000)
y_val=np.full(5000,y_mean)
plt.plot(x_val,y_val)#esto nose bien para que lo hago

x_mean=np.mean(franja_x)




#Y esto para todos los cosos y dsps se puede hacer algo integral


#tomemos el fotopico como el maximo y asignemosle un error comparado con el ancho de la campana
fotopico_1= fotopico
err_fot=err
#voltajedefot= aaaa














#%%
#ACA SE PUEDE HACER CODIGO PARA COMBINAR DATOS
#veo ambos juntos
bismuto1 = np.loadtxt('datobismuto4largo.txt')
#cesio1= np.loadtxt('datoscesio1.txt')
cesio1=datoscomb

plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(0,1.2,50)
axs.hist(bismuto1,bins = bins)
axs.set_title("Histograma picos")
axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")
axs.hist(cesio1,bins = bins)
axs.set_title("Histograma picos")
axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")

plt.figure()
hist_ces = axs.hist(datoscomb,bins=bins)
histx_ces= np.array(hist_ces[0])
histx_ces =  np.insert(histx_ces,1,0,axis=None)
histy_ces= np.array(hist_ces[1])
plt.plot(histy_ces,histx_ces)
plt.yscale("log")


hist_bis = axs.hist(bismuto1,bins=bins)
histx_bis= np.array(hist_bis[0])
histx_bis =  np.insert(histx_bis,1,0,axis=None)
histy_bis= np.array(hist_bis[1])

plt.plot(histy_bis,histx_bis)




#%%
#para sacar los fotopicos cualitativamente de los graficos que son un solo txt(mismo cod que antes)
datoscomb=np.loadtxt('datobismuto4largo.txt')

plt.close('all')

plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(0,1.2,75)
axs.hist(datoscomb,bins = bins)
axs.set_title("Histograma picos")
axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")

hist = axs.hist(datoscomb,bins=bins)
histx= np.array(hist[0])
histx =  np.insert(histx,1,0,axis=None)
histy= np.array(hist[1])
plt.figure()
plt.plot(histy,histx)
plt.yscale("log")
histcomp= [histx,histy]
xmax,ymax = scipy.signal.find_peaks(histx,height = 1000) #se va a cagar los xmax no sirven para nada, modificar la tol aca

#hago que xmax corresponda a los graficos
xmax_bien= []
ymax_bien= []
for i in range(len(xmax)):
    i=xmax[i]
    p=histy[i]
    q=histx[i]
    xmax_bien.append(p)
    ymax_bien.append(q)
    

plt.scatter(xmax_bien,ymax_bien)

#VEAMOS cuaL ES EL ancho cualitativamente
busqueda= 2

fotopico= xmax_bien[3] #FIJARSE ACA QUE SEA EL FOTOPCIO
#fotopico_cs
#fotopico_bis
#fotopico_asd
#fotopico_bsd
#fotopico_DSAD
err_fot=[]
#err_foto_cs =[]
#err_foto_bis=[]
#err_foto_asd=[]
#err_foto_bsd={}
#err_foto_DSAD={}

j=0
while j < len(histy):
    if histy[j] == fotopico:
        j_primab= j-busqueda
        j_primaf=j+busqueda
        zonax=histy[j_primab]
        zonaxf= histy[j_primaf]
        plt.scatter(zonax,histx[j_primab],c='#2ca02c')
        plt.scatter(zonaxf,histx[j_primaf],c='#2ca02c')
        j=j+1
        
        franja_x=[]
        franja_y=[]
        while j_primab<=j_primaf:
            franja_x.append(histy[j_primab])
            franja_y.append(histx[j_primab])
            j_primab=j_primab +1
        plt.figure()
        plt.plot(franja_x,franja_y, c='#d62728')
        err_fot.append(franja_x[busqueda]-franja_x[0])
    else:
        j=j+1

distribution=franja_x
weights=franja_y
def weighted_average_m1(distribution, weights):
  
    numerator = sum([distribution[i]*weights[i] for i in range(len(distribution))])
    denominator = sum(weights)
    
    return round(numerator/denominator,2)
print(weighted_average_m1(distribution, weights))
franja_x_array=np.array(franja_x)
franja_y_array=np.array(franja_y)
#calculate weighted standard deviation
print(DescrStatsW(distribution, weights=weights, ddof=1).std)
#ploteemos el histo del fotopcio
plt.figure()
fig, axs = plt.subplots()
bins = np.linspace(franja_x[0],franja_x[len(franja_x)-1],len(franja_x))
axs.hist(franja_y,bins = bins)
axs.set_title("Histograma picos")
#axs.set_yscale("log")
axs.set_xlabel("Voltaje (v)")
axs.set_ylabel("Cuentas")

#bueno aca chicos tenemos que buscar una forma de determinar el fotopico
#para eso tiene que haber dos formas, una burda que seria tomar el pico detectado y ponerle un error de +- el ancho que se ve en el grafico
#y la otra seria mediante un ajuste o magia estadistica determinar un maximo y su error
#%%
#ACA ME Guardo los fotopicos FIJARSE DE IR DESCOMENTANDO Y COMENTANDO
#BISMUTO
#fot_pic1_bis= fotopico
#err_fot_pic1_bis=err_fot
fot_pic1_bis_real= 12.7#en MeV
#fot_pic2_bis= fotopico
#err_fot_pic2_bis=err_fot
fot_pic2_bis_real= 72.8#en MeV

#CESIO
fot_pic1_ces= fotopico
err_fot_pic1_ces=err_fot
fot_pic1_ces_real=32.1 #Mev

#me armo una lista con todos los fotopicos
fotopicos=[]
fotopicos.append(0)
fotopicos.append(fot_pic1_bis)
fotopicos.append(fot_pic2_bis)
fotopicos.append(fot_pic1_ces)

err_fotopicos=[]
err_fotopicos.append(err_fot_pic1_bis)
err_fotopicos.append(err_fot_pic2_bis)
err_fotopicos.append(err_fot_pic1_ces)

valores_reales=[]
valores_reales.append(0)
valores_reales.append(fot_pic1_bis_real)
valores_reales.append(fot_pic2_bis_real)
valores_reales.append(fot_pic1_ces_real)


plt.figure()
plt.scatter(fotopicos,valores_reales)




















#%%
datos_dict = {}
#%%
gain_value = 0.1
#%%
## Medicion continua
#PARA HACER MEDICIONES GUARDANDO TODOS LOS DATOS (NO SOLO PICOS)
#(la usamos para tener una medicion corta con todos los datos, es muy pesado asi que para sacar para tiempos largos usamos otra adquisición abajo
fs = 125000 #Frecuencia de muestreo máxima permitida por NIDAQ
duracion=1
 #o mas chiquito todavia
datos_canal1 = []
datos_canal2 = []

task = nidaqmx.Task()
with nidaqmx.Task() as task:
    modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
    task.ai_channels.add_ai_voltage_chan("Dev5/ai1", terminal_config = modo,
                                         max_val=10,min_val=0) #deberían estar bien con estos límites
    task.ai_channels.add_ai_voltage_chan("Dev5/ai2", terminal_config = modo,
                                         max_val=10,min_val=0) #deberían estar bien con estos límites
    #pero capaz haga falta subirlo, idk
    task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS) 
    task.start()
    t0 = time.time()
    total = 0
    T = []
    while total < (fs*duracion):
        time.sleep(0.1)
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        datos_canal1 += datos[0]
        datos_canal2 += datos[1]
        total += len(datos[0])
        t = time.time() - t0
        T.append(t)
#        print(total, fs*duracion)
#        print("%2.3fs %d %d %2.3f" % (t, len(datos), total, total/(t)))    

datos_canal1 = np.array(datos_canal1)
datos_canal2 = np.array(datos_canal2)

datos_dict[f"gain {gain_value}"] = datos_canal1
datos_dict[f"gain {gain_value} fuente"] = datos_canal2


os.chdir(r"C:\Users\Publico\Desktop\l5 g1 verano 23")

np.savez("datos", datos=datos)
np.savez(f"(T) t {gain_value}", T)
np.savez(f"(total) len {gain_value}", total)
#%%
file_name = "ClK.csv"


#%%
df_datos = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in datos_dict.items() ]))
#df_datos.to_csv(file_name)
#%%
#df_datos = pd.read_csv("shaping 8 2.csv")
medicion_actual = df_datos["gain 0.1"]
bins = int(len(medicion_actual)/1600)
peaks, _ = scipy.signal.find_peaks(medicion_actual,height = 0.1)
#%%
fig, axs = plt.subplots(1,2)
axs[0].hist(medicion_actual,bins = bins)
axs[0].hist(medicion_actual[peaks],bins = bins)
axs[0].set_title("Histograma completo")
axs[0].set_yscale("log")
axs[0].set_xlabel("Voltaje (v)")
axs[0].set_ylabel("Cuentas")
#axs[0].axis([1, 11, np.min(medicion_actual),np.max(medicion_actual)])

axs[1].hist(medicion_actual[peaks],bins = bins)
#axs[1].set_ylim(0,2000)
axs[1].set_title("Buscando picos")
axs[1].set_yscale("log")
axs[1].set_xlabel("Voltaje (v)")
axs[1].set_ylabel("Cuentas")
#axs[1].axis([1, 11, np.min(medicion_actual),np.max(medicion_actual)])
plt.show()

 #%%
peaks, _ = scipy.signal.find_peaks(datos_canal1,height = 0.01)

#plt.figure()
#plt.plot(datos, marker = "o")
#for i in range(len(peaks)):
#    plt.plot(peaks[i], datos[peaks[i]], color = "magenta", marker="o") 
#    



plt.figure()
#plt.plot(datos_canal1) #, marker = "o")
plt.ylabel("Datos" )
plt.yscale("log")
plt.xlabel("Voltaje [V]")
plt.hist(medicion_actual[peaks],1000, color = "magenta", linewidth=0, label = "Picos")
plt.legend() 


#%%
#llevarse una medicion corta para tener un grafico de lo que se lee el daq

file = "Cs60sAMP2 (fmt .5e).txt"
#np.savez(file, datos=datos, fmt='%.5e')

#%%
#medicion que solo guarda los máximos: 
#USAR ESTA para medir con tiempos largos (primero probar con la del fondo que capaz anda)
    
## Medicion continua
fs = 100000 #Frecuencia de muestreo (maxima permitida por nidaq
duracion = 10 #variar de acuerdo a la muestra, con tal de tener varios picos
peaks = np.zeros(int(fs*duracion)) #acá me guardo los valores de los picos
loc_peaks =  np.zeros(fs*duracion) #acá los tiempos de cada pico (para hacer analisis estadistico)

t_oops= [] #me appendeo acá los tiempos en los cuales la compu falla

task = nidaqmx.Task()
with nidaqmx.Task() as task:
    modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
    task.ai_channels.add_ai_voltage_chan("Dev5/ai1", terminal_config = modo,max_val=10,min_val=0)
    task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS) #el modo continuous no permite fijar una
    #cantidad de datos a medir, el tipo lee nomás hasta q lo pares
    task.start()
    t0 = time.time()
    tf = t0
    total = 0  #acá me guardo la cantidad total de datos leídos
    cant_peaks = 0    #acá la cantidad total de picos
    while total< (fs*duracion):#-100000):  #no estoy seguro por que estaba el -100000, si no funca probar ponerlo
        time.sleep(0.1)  #en este tiempo permito que lea datos con modo continuous. si lee más de 100000 datos se llena la memoria buffer de la compu
        t_i = tf #tiempo inicial de cada iteracion
        tf = time.time()
        deltat = tf-t_i
        print(tf-t0, end="\n")
        if deltat>0.18:  #nos paso que la compu entre iteraciones realizaba otras tareas q nada que ver, y se llenaba así la memoria buffer 
        #y tiraba error. entonces si la dif. de tiempo>0.18 s, ya sabemos que la memoria va a estar llena y reinicio el task antes de que tire error
            print('oops i did it again i played with your heart')
            task.stop()
            task.start()
            t_oops.append(tf)
        #print("%2.3fs %d %d %2.3f" % (tf-t0, len(datos), total, total/(tf-t0)), resol_daq, cant_peaks)             
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE) #esto te guarda los datos que estén en la memoria
        datos = np.asarray(datos
                           )#capaz haya que agregarle un factor menos, nos paso a nosotros
        
        if len(datos)>1: #a veces no leía datos y habia que agregar esto
            resol_daq = np.min(np.diff(np.unique(datos))) #esto es para graficar despues
            peaks_indx, _ = scipy.signal.find_peaks(datos,height = 0.05)  #fijense si hace falta cambiar los parametros del findpeaks
            peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=datos[peaks_indx]   #te guarda todos los peaks que va midiendo
            loc_peaks[cant_peaks:(cant_peaks+len(peaks_indx))] = (peaks_indx+total)/fs  #guarda los tiempos de cada pico
            cant_peaks = cant_peaks +len(peaks_indx)
#            tiempos = np.linspace(t_i, tf, num = len(datos))   #no estoy seguro por que esta muteado esto, creeria que tmb funciona para guardar tiempos
#            loc_peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=tiempos[peaks_indx]

        else:
            resol_daq = 0 
        total = total + len(datos)
    
peaks = peaks[:cant_peaks]  #recorta el array para descartar los que quedaron vacíos
loc_peaks = loc_peaks[:cant_peaks]


#%% MEDICION DE LA REGION DEL FOTOPICO
"""
#medicion que solo guarda los máximos: 
#USAR ESTA para medir con tiempos largos (primero probar con la del fondo que capaz anda)
    
## Medicion continua
fs = 80000 #Frecuencia de muestreo (maxima permitida por nidaq
duracion=60*10 #variar de acuerdo a la muestra, con tal de tener varios picos
peaks = np.zeros(int(fs*duracion)) #acá me guardo los valores de los picos
loc_peaks =  np.zeros(fs*duracion) #acá los tiempos de cada pico (para hacer analisis estadistico)
t_oops= [] #me appendeo acá los tiempos en los cuales la compu falla
lims = (2, 10) # SETEAR REGION
task = nidaqmx.Task()
with nidaqmx.Task() as task:
    modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
    task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo,max_val=lims[1],min_val=lims[0]-0.1)
    task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS) #el modo continuous no permite fijar una
    #cantidad de datos a medir, el tipo lee nomás hasta q lo pares
    task.start()
    t0 = time.time()
    tf = t0
    total = 0  #acá me guardo la cantidad total de datos leídos
    cant_peaks = 0    #acá la cantidad total de picos
    while total< (fs*duracion):#-100000):  #no estoy seguro por que estaba el -100000, si no funca probar ponerlo
        time.sleep(0.1)  #en este tiempo permito que lea datos con modo continuous. si lee más de 100000 datos se llena la memoria buffer de la compu
        t_i = tf #tiempo inicial de cada iteracion
        tf = time.time()
        deltat = tf-t_i
        print(tf-t0, end="\n")
        if deltat>0.18:  #nos paso que la compu entre iteraciones realizaba otras tareas q nada que ver, y se llenaba así la memoria buffer 
        #y tiraba error. entonces si la dif. de tiempo>0.18 s, ya sabemos que la memoria va a estar llena y reinicio el task antes de que tire error
            print('oops i did it again i played with your heart')
            task.stop()
            task.start()
            t_oops.append(tf)
        #print("%2.3fs %d %d %2.3f" % (tf-t0, len(datos), total, total/(tf-t0)), resol_daq, cant_peaks)             
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE) #esto te guarda los datos que estén en la memoria
        datos = np.asarray(datos)#capaz haya que agregarle un factor menos, nos paso a nosotros
        
        if len(datos)>1: #a veces no leía datos y habia que agregar esto
            resol_daq = np.min(np.diff(np.unique(datos))) #esto es para graficar despues
            peaks_indx, _ = scipy.signal.find_peaks(datos,height = (lims[0]+1,lims[1]-1))  #fijense si hace falta cambiar los parametros del findpeaks
            peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=datos[peaks_indx]   #te guarda todos los peaks que va midiendo
            loc_peaks[cant_peaks:(cant_peaks+len(peaks_indx))] = (peaks_indx+total)/fs  #guarda los tiempos de cada pico
            cant_peaks = cant_peaks +len(peaks_indx)
#            tiempos = np.linspace(t_i, tf, num = len(datos))   #no estoy seguro por que esta muteado esto, creeria que tmb funciona para guardar tiempos
#            loc_peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=tiempos[peaks_indx]
        else:
            resol_daq = 0 
        total = total + len(datos)
    
peaks = peaks[:cant_peaks]  #recorta el array para descartar los que quedaron vacíos
loc_peaks = loc_peaks[:cant_peaks]
"""
#%%
plt.figure()
plt.yscale("log")
n,bins,_ = plt.hist(peaks, bins=800)
#plt.xlim(0.1,10)
#plt.ylim(0,100)
#%%
#histograma estandar
csfont = {"fontname" : "Helvetica"}
plt.figure(figsize = (8,6))
plt.plot(bins[:-1],n, ".", ms=2)
plt.xticks(fontsize = 12)
plt.xticks( fontsize = 12)
plt.xlabel("Voltaje [V]", fontsize = 15)
plt.ylabel("Cuentas", fontsize = 15)
plt.yscale("log")
#plt.xscale("log")

#plt.xlim(0.1,10)
#plt.ylim(100,10000)
plt.show()

#%%

t_oops_array = np.zeros(len(peaks))
t_oops_array[0:len(t_oops)]= t_oops

os.chdir(r"D:\L5 Grupo 3 Buono-Colomb-Gaburri\Nuclearrrr\Mediciones\18-11\meds largas\Plaqita cobre")
data= np.column_stack([peaks, loc_peaks, t_oops_array]) #aca stackeo las columnas a exportar
file = open("Cs137 - CG30 - G05 - combineitor mezcli.txt", "w")
np.savetxt(file, data, header= 'Picos [V], Tiempos [s], T_oops')
file.close()


#%%
#aca vamos guardando los histogramas de cada fuente,
#descomentando solo la que estamos midiendo

#conviene usar un multiplo de la minima
#resolucion de voltaje para el ancho de bines, para evitar
#cosas raras con aliasing, dobles lineas, etc.
volt_range = np.arange(0, 10, resol_daq * 50)



#cs_hist =  np.histogram(peaks, volt_range)
# cs_2_hist =  np.histogram(peaks, volt_range)
# cs_3_hist =  np.histogram(peaks, volt_range)
#bi_hist =  np.histogram(peaks, volt_range)
#ba_hist =  np.histogram(peaks, volt_range)
# co_1_hist =  np.histogram(peaks, volt_range)
#na_1_hist =  np.histogram(peaks, volt_range)
#na_2_hist =  np.histogram(peaks, volt_range)
#%%
#aca vamos ploteando todo junto a medida que vamos midiendp

csfont = {"fontname" : "Helvetica"}
plt.figure(figsize = (10,8))
#plt.plot(cs_hist[1][:-1], cs_hist[0],'o', label = 'Cs')
## plt.plot(cs_2_hist[1][:-1], cs_2_hist[0],'o', label = 'Cs 2')
#plt.plot(bi_hist[1][:-1], bi_hist[0],'o', label = 'Bi ')
#plt.plot(ba_hist[1][:-1], ba_hist[0],'o', label = 'Ba')
## plt.plot(co_1_hist[1][:-1], co_1_hist[0],'o', label = 'Ba azul 2')
#plt.plot(na_1_hist[1][:-1], na_1_hist[0],'o', label = 'Na1')
plt.plot(na_2_hist[1][:-1], na_2_hist[0],'o', label = 'Na2')
plt.xticks(fontsize = 12)
plt.yticks( fontsize = 12)
plt.xlabel("Voltaje [V]", fontsize = 15)
plt.ylabel("Cuentas", fontsize = 15)
plt.legend(fontsize = 18)


#%%
#si veo que se puede apreciar el fotopico en el grafico uso esto para determinar 
#el voltaje correspondiente

def fotopico(hist, i):
    array = hist[0][i:-1]
    m = np.max(array)
    l = (hist[0].tolist()).index(m)
    pico = hist[1][l]
    return pico
#fijar i como el voltaje en V (*30 por cantidad de bins) a partir del cual el fotopico es el max
pico_cs_1 = fotopico(cs_1_hist, i)
# pico_cs_2 = fotopico(cs_2_hist, i)
# pico_cs_3 = fotopico(cs_3_hist, i)
# pico_bi = fotopico(bi_hist, i)
# pico_ba_azul = fotopico(ba_azul_hist, i)
# pico_co = fotopico(co_hist, i)
# pico_na = fotopico(na_hist, i)

#%%
#aca hago el ajuste de calibración comparando con los datos de la tabla


def recta(x, a, b):
    y = a * x + b
    return y


picos_x = np.array([pico_cs_1, pico_bi, pico_ba_azul, pico_na])#poner picos que se vean bien
picos_y = np.array()#poner valores de la tabla que crean que se correspondan])      

    
#comentario: en casa vi que estos picos se ajustaban bien con estos valores tabulados    
# picos_x = np.array([pico_ba_azul_2, pico_bi_207, pico_2_bi])
# picos_y = np.array([383.017, 569.702, 1063])     
#Resultaron en: slope = 306, oo= -3.7


    
popt, pcov = curve_fit(recta, picos_x, picos_y)    
perr = np.sqrt(np.diag(pcov))

slope = popt[0]
err_slope = perr[0]
oo = popt[1] #ordenada al origen
err_oo = perr[1]

print(f'Pendiente: ', slope, '+-', err_slope)
print(f'Ordenada al origen: ', oo, '+-', err_oo)


ajuste = recta(picos_x, popt[0], popt[1])


csfont = {"fontname" : "Helvetica"}
plt.figure(figsize = (10,8))
plt.plot(picos_x, picos_y, 'o')
plt.plot(picos_x, ajuste, label = 'Ajuste lineal')
plt.xticks(fontsize = 12)
plt.yticks( fontsize = 12)
plt.xlabel("Fotopicos experimentales", fontsize = 15)
plt.ylabel("Fotopicos tabulados", fontsize = 15)
plt.legend(fontsize = 18)


#%%
#ACA HACER ALGO CON EL BORDE DE COMPTON SI SE PUEDE VER EN 3 O MAS MUESTRAS

#%%
#ANALISIS ESTADISTICO
#queremos ver la distrubicion de la cantidad de energias del fotopico
#en diferentes intervalos de tiempo para ver si se puede modelar con una poisson.
#Medimos por 10 minutos y luego subdividimos el tiempo total en intervalos más pequenos.
#Si queda tiempo podemos variar la fuente y la distancia entre el detector y la fuente.
#mientras mas lejos deberíamos ver un lambda más pequeno.
#(podes medir las distancias también)


#medicion que solo guarda los máximos:
#ACLARACION: en este caso debemos guardar tambien la ubicacion (temporal) de los maximos    


## Medicion continua
fs = 300000 #Frecuencia de muestreo
duracion=60*5 #primero probar con una cortita por las dudas
peaks = np.zeros(int(fs*duracion)) #claramente va a haber muchos menos pero por las
loc_peaks =  np.zeros(fs*duracion)



task = nidaqmx.Task()
with nidaqmx.Task() as task:
    modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
    task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo,max_val=10,min_val=-10)
    task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
    task.start()
    t0 = time.time()
    total = 0
    cant_peaks = 0
    t1 = t0    
    while total< (fs*duracion):#-100000):   esto creo que estaba para que no se superara el tamano de datos pero not sure
        time.sleep(0.1)
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE) 
        datos = np.asarray(datos) #daba los datos con los picos negativos
        peaks_indx, _ = scipy.signal.find_peaks(datos,height = 0.05)#setear umbral para q queden los cercanos al fotopico nomas
        peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=datos[peaks_indx]
        # cant_peaks = cant_peaks +len(peaks_indx)
        t_i = t1 #tiempo inicial de cada iteracion
        total = total + len(datos)
        t1 = time.time()
        tiempos = np.linspace(t_i, t1, num = len(datos))
        loc_peaks[cant_peaks:(cant_peaks+len(peaks_indx))]=tiempos[peaks_indx]
        cant_peaks = cant_peaks +len(peaks_indx)
        print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)), cant_peaks)    
    
peaks = peaks[:cant_peaks]
loc_peaks = loc_peaks[:cant_peaks]

#%%
plt.close('all')
#histograma estandar
csfont = {"fontname" : "Helvetica"}
plt.figure(figsize = (10,8))
plt.hist(peaks, bins=100, color = "maroon")
plt.xticks(fontsize = 12)
plt.xticks( fontsize = 12)
plt.xlabel("Voltaje [V]", fontsize = 15)
plt.ylabel("Cuentas", fontsize = 15)

#%%
plt.close('all')
loc_peaks = np.array( loc_peaks ) - np.ones(len(loc_peaks))*t0
plt.plot(loc_peaks,peaks)

#%%
#llevarse una medicion corta para tener un grafico de lo que se lee el daq

file = open("Na02EST10mVOL1.txt", "w")

data= np.column_stack([peaks, loc_peaks]) #aca stackeo las columnas a exportar

np.savetxt(file, data, fmt='%.5e', header= 'Picos [V], Tiempos [s]', delimiter = ' ')
file.close()

#%%
datos=
