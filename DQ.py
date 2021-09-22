import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import uniform
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
import csv
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import seaborn as sn
import requests
import json
import haversine as hs
import wx
#holaaaa
#HOLA2


#Read Data from February
#header_CC=["codigoSerial", "fecha", "hora", "fechaHora", "temperatura", "humedad_relativa", "pm1_df", "pm10_df", "pm25_df", "pm1_nova", "pm10_nova", "pm25_nova", "calidad_temperatura", "calidad_humedad_relativa", "calidad_pm1_df", "calidad_pm10_df", "calidad_pm25_df", "calidad_pm1_nova", "calidad_pm10_nova", "calidad_pm25_nova"]
#datatypes_CC={"codigoSerial":np.uint16, "temperatura":np.float16, "humedad_relativa":np.float16, "pm1_df":np.float32, "pm10_df":np.float32, "pm25_df":np.float32, "pm1_nova":np.float32, "pm10_nova":np.float32, "pm25_nova":np.float32}
#df_CC = pd.read_csv("C:/Users/julio/Documents/UDEA/Maestría/DQ in IOT/Datasets/SIATA_CS/SplitDatosCC/Samples/"+"February.csv", header=None, names=header_CC, usecols=header_CC , dtype=datatypes_CC,parse_dates=["fecha","hora","fechaHora"])

#Data includes January, February and March
#header_SS=["Fecha_Hora","codigoSerial","pm25","calidad_pm25","pm10","calidad_pm10"]
#datatypes_SS={"codigoSerial":np.uint16,"pm25":np.float32,"pm10":np.float32}
#df_SS = pd.read_csv("C:/Users/julio/Documents/UDEA/Maestría/DQ in IOT/Datasets/SIATA Stations/PM/"+"SS_PM.csv", header=None,names=header_SS, usecols=header_SS , dtype=datatypes_SS,parse_dates=["Fecha_Hora"])


#grouped=df_CC.groupby(df_CC.codigoSerial)
#CC={}
#print("Citizen Scientist: ", sorted(list(df_CC.codigoSerial.unique())))
#for i in df_CC.codigoSerial.unique():
#    CC[i] = grouped.get_group(i).sort_values(by=['fechaHora'],ignore_index=True)
#del df_CC

#grouped=df_SS.groupby(df_SS.codigoSerial)
#SS={}
#print("Siata Stations: ", list(df_SS.codigoSerial.unique()))
#for j in df_SS.codigoSerial.unique():
#    SS[j] = grouped.get_group(j).sort_values(by=['Fecha_Hora'],ignore_index=True)
#del df_SS
#del grouped

#datatypesDistances={"codigoSerial_CC":np.uint16,"codigoSerial_ES":np.uint16,"Distancia_a_ES":np.float16,"codigoSerial_ES2":np.uint16}
#Distances = pd.read_csv("C:/Users/julio/Documents/UDEA/Maestría/DQ in IOT/Datasets/Distances and positions/Distancias_2.csv", header=0, dtype=datatypesDistances,index_col="codigoSerial_CC")


#start_time ="2020-02-10 00:00:00"
#end_time   ="2020-02-10 05:59:00"

def clean_data(node, CC, start_time, end_time):
    
    #define a time windows of the data to be analyzed/cleaned
    df_window=CC[node][(CC[node]['fechaHora'] >= start_time) & (CC[node]['fechaHora'] <= end_time)]
    #Remove outliers that are out of range, from documentation both nova and df range of measurements are [0,999]
    df_window=df_window.copy()
    df_window.loc[df_window["pm25_nova"]>999,"pm25_nova"]=np.nan
    df_window.loc[df_window["pm25_nova"]<0,"pm25_nova"]=np.nan
    df_window.loc[df_window["pm25_df"]>999,"pm25_df"]=np.nan
    df_window.loc[df_window["pm25_df"]<0,"pm25_df"]=np.nan
    
    #Remove data above the whiskers of the boxplot: i.e. anomaly data
    Q1 = df_window['pm25_df'].quantile(0.25)
    Q3 = df_window['pm25_df'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_window.loc[df_window["pm25_df"]>=Q3 + 1.5 *IQR,"pm25_df"]=np.nan
    
    Q1 = df_window['pm25_nova'].quantile(0.25)
    Q3 = df_window['pm25_nova'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_window.loc[df_window["pm25_nova"]>=Q3 + 1.5 *IQR,"pm25_nova"]=np.nan
    
    return df_window
    
def precision(node, df_window):
    #Hourly standard deviation
    df_window['pm25_nova_pre']=np.nan
    df_window['pm25_df_pre']=np.nan
    for ts in df_window['fechaHora']:
        if ts==ts.ceil('60min'):
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < (ts+timedelta(minutes = 1)).ceil('60min'))]# Next ONE HOUR WINDOW
            
        else:
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < ts.ceil('60min'))]#Current ONE HOUR window
        #print("Timestamp: ",ts,", Floor:",ts.floor('60min'),", Ceil:",ts.ceil('60min'),window['pm25_nova'].mean())
        #print(window['pm25_nova'])        
        df_window.loc[df_window["fechaHora"]==ts,'pm25_nova_pre']=1-(window['pm25_nova'].std()/window['pm25_nova'].mean())# 1-Coefficient of Variation (std/mean)
        df_window.loc[df_window["fechaHora"]==ts,'pm25_df_pre']=1-(window['pm25_df'].std()/window['pm25_df'].mean())# 1-Coefficient of Variation (std/mean)
        
    #del window
    prec_df=df_window.pm25_df_pre.mean()#Average precision of the whole node for DF
    prec_nova=df_window.pm25_nova_pre.mean()  #Average precision of the whole node for NOVA  
    preci_dict={"codigoSerial":node,"precision_df":prec_df,"precision_nova":prec_nova}
    #print("%d. Nube: %d, Overall relative (Precision) Standard Deviation."%(contador,nube))
    del prec_df
    del prec_nova
    return preci_dict


#df_preci=df_preci.append(precision(node, df_window), ignore_index = True)
#df_preci.to_csv("C:/Users/julio/Documents/UDEA/Maestría/DQ in IOT/Datasets/DQ_February/df_preci.csv",index=False)


def uncertainty(node, df_window):
    #Hourly standard deviation

    df_window['pm25_unc']=np.nan
    for ts in df_window['fechaHora']:
        if ts==ts.ceil('60min'):
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < (ts+timedelta(minutes = 1)).ceil('60min'))]# Next ONE HOUR WINDOW
            
        else:
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < ts.ceil('60min'))]#Current ONE HOUR window
        #print("Timestamp: ",ts,", Floor:",ts.floor('60min'),", Ceil:",ts.ceil('60min'),window['pm25_nova'].mean())
        #print(window['pm25_nova'])        
        df_window.loc[df_window["fechaHora"]==ts,'pm25_unc']=\
        1-np.sqrt((window.pm25_df-window.pm25_nova).pow(2).mean()/2)/((window.pm25_df+window.pm25_nova).mean()/2)
        
    #del window
    
    uncer_df=df_window.pm25_unc.mean()#Average uncertainty of the whole node
    uncer_dict={"codigoSerial":node,"uncertainty":uncer_df}
    del uncer_df
    
    return uncer_dict


#df_preci=df_preci.append(precision(node, df_window), ignore_index = True)
#df_preci.to_csv("C:/Users/julio/Documents/UDEA/Maestría/DQ in IOT/Datasets/DQ_February/df_preci.csv",index=False)


def accuracy(node, df_window,Distances,SS):
    #Hourly mean
    df_window['pm25_nova_ave']=np.nan
    df_window['pm25_df_ave']=np.nan    
    df_window['alpha_df']=np.nan   
    df_window['alpha_nova']=np.nan 
    
    
    for ts in df_window['fechaHora']:
        if ts==ts.ceil('60min'):
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < (ts+timedelta(minutes = 1)).ceil('60min'))]# Next ONE HOUR WINDOW
            
        else:
            window=df_window[(df_window['fechaHora'] >= ts.floor('60min')) & (df_window['fechaHora'] < ts.ceil('60min'))]#Current ONE HOUR window
        #print("Timestamp: ",ts,", Floor:",ts.floor('60min'),", Ceil:",ts.ceil('60min'),window['pm25_nova'].mean())
        #print(window['pm25_nova'])        
        df_window.loc[df_window["fechaHora"]==ts,'pm25_nova_ave']=window['pm25_nova'].mean()
        df_window.loc[df_window["fechaHora"]==ts,'pm25_df_ave']=window['pm25_df'].mean()

    Closest_Station=Distances.codigoSerial_ES.loc[node]    
    Closest_Station2=Distances.codigoSerial_ES2.loc[node] 
    if Closest_Station in SS.keys():
        #Clean values out of range
        SS[Closest_Station].loc[SS[Closest_Station]["pm25"]<=0,"pm25"]=np.nan
        for time in df_window.fechaHora:
            
            idx=SS[Closest_Station].Fecha_Hora.searchsorted(time,side="right")
            #print(idx, SS[Closest_Station].Fecha_Hora.loc[idx], time)
            v=SS[Closest_Station].loc[idx,"pm25"]
            df_window.loc[df_window.fechaHora == time,"v_pm25"]=v
            vm=df_window.loc[(df_window.fechaHora == time),"pm25_df_ave"]
            #print(time," : ",vm.values[0],"________",SS[Closest_Station].Fecha_Hora.loc[idx]," : ",v)
            #print(v,"<->", vm.values[0])
            df_window.loc[df_window.fechaHora == time,"alpha_df"]=  max(0,1-abs(vm.values[0]-v)/v)
            vm=df_window.loc[(df_window.fechaHora == time),"pm25_nova_ave"]
            df_window.loc[df_window.fechaHora == time,"alpha_nova"]=max(0,1-abs(vm.values[0]-v)/v)
        
        accu_dict={'codigoSerial':node, 'dist':Distances.loc[node,"Distancia_a_ES"], 'acc_df':df_window.alpha_df.mean(), 'acc_nova':df_window.alpha_nova.mean()}
    
    elif Closest_Station2 in SS.keys():
        #Clean values out of range
        SS[Closest_Station2].loc[SS[Closest_Station2]["pm25"]<=0,"pm25"]=np.nan
        for time in df_window.fechaHora:
            
            idx=SS[Closest_Station2].Fecha_Hora.searchsorted(time,side="right")
            #print(idx, SS[Closest_Station].Fecha_Hora.loc[idx], time)
            v=SS[Closest_Station2].loc[idx,"pm25"]
            df_window.loc[df_window.fechaHora == time,"v_pm25"]=v
            vm=df_window.loc[(df_window.fechaHora == time),"pm25_df_ave"]
            #print(time," : ",vm.values[0],"________",SS[Closest_Station].Fecha_Hora.loc[idx]," : ",v)
            #print(v, vm)
            df_window.loc[df_window.fechaHora == time,"alpha_df"]=  max(0,1-abs(vm.values[0]-v)/v)
            vm=df_window.loc[(df_window.fechaHora == time),"pm25_nova_ave"]
            df_window.loc[df_window.fechaHora == time,"alpha_nova"]=max(0,1-abs(vm.values[0]-v)/v)
        
        accu_dict={'codigoSerial':node, 'dist':Distances.loc[node,"Distancia_a_ES"], 'acc_df':df_window.alpha_df.mean(), 'acc_nova':df_window.alpha_nova.mean()}
        
    else:
        accu_dict={'codigoSerial':node, 'dist':np.nan, 'acc_df':np.nan, 'acc_nova':np.nan}
    return accu_dict


def concordance(node, df_window,Distances,SS):
    df_window['v_pm25']=np.nan 
    
    Closest_Station=Distances.codigoSerial_ES.loc[node]    
    if Closest_Station in SS.keys():
        #Clean values out of range
        SS[Closest_Station].loc[SS[Closest_Station]["pm25"]<=0,"pm25"]=np.nan
        for time in df_window.fechaHora:
            
            idx=SS[Closest_Station].Fecha_Hora.searchsorted(time,side="right")
            #print(idx, SS[Closest_Station].Fecha_Hora.loc[idx], time)
            v=SS[Closest_Station].loc[idx,"pm25"]
            df_window.loc[df_window.fechaHora == time,"v_pm25"]=v

    #Comparability / Concordance
    corr_df = df_window.loc[:,["pm25_df","pm25_nova","v_pm25","temperatura","humedad_relativa"]].corr().iloc[0].abs()
    corr_nova = df_window.loc[:,["pm25_df","pm25_nova","v_pm25","temperatura","humedad_relativa"]].corr().iloc[1].abs()
    conco_dict={"codigoSerial":node,"concordance_df_nova":corr_df.pm25_nova,
                "concordance_df_siata":corr_df.v_pm25,"concordance_df_hum":corr_df.humedad_relativa,"concordance_df_temp":corr_df.temperatura,
                "concordance_nova_siata":corr_nova.v_pm25,"concordance_nova_hum":corr_nova.humedad_relativa,"concordance_nova_temp":corr_nova.temperatura}
    return conco_dict
    
    
def completeness(node, df_window, start_time, end_time):
    #ref_date_range = pd.date_range(inicio, fin, freq='1Min')
    #ref_date_range = pd.DataFrame(ref_date_range,columns=["ref_fechaHora"])
    ref_date_range = pd.DataFrame(pd.date_range(start_time, end_time, freq='1Min'),columns=["ref_fechaHora"])

    #Check for any missing date
    missing_dates = ref_date_range.loc[~ref_date_range.ref_fechaHora.isin(df_window.fechaHora),"ref_fechaHora"]

    #Add missing date rows
    for missing in missing_dates:
        df_window=df_window.append({"codigoSerial":node,"fechaHora":missing}, ignore_index = True)
    
    #Check for any missing date
    #missing_dates = ref_date_range.loc[~ref_date_range.ref_fechaHora.isin(df_window.fechaHora),"ref_fechaHora"]
    
    #Check for missing data
    missing_data_df=np.count_nonzero(np.isnan(df_window['pm25_df']))
    missing_data_nova=np.count_nonzero(np.isnan(df_window['pm25_nova']))
    comp_df=(1-missing_data_df/np.size(df_window.pm25_df))
    comp_nova=(1-missing_data_nova/np.size(df_window.pm25_nova))
    
    comp_dict={"codigoSerial":node,"completeness_df":comp_df,"completeness_nova":comp_nova}
    return comp_dict


contador=0
def eval_dq(arguments):
    nodes=arguments[0]
    CC=arguments[1]
    SS=arguments[2]
    Distances=arguments[3]
    start_time=arguments[4]
    end_time=arguments[5]

    #contador+=1
    #CC[nube]["v_pm25"] = np.nan
    #CC[nube]["alpha_df"] = np.nan
    #CC[nube]["alpha_nova"] = np.nan
    #del df_window
    
    df_window=clean_data(nodes, CC, start_time, end_time)
              
    #PRECISION
    #df_preci=df_preci.append(precision(nodes, df_window), ignore_index = True)
    df_preci=precision(nodes, df_window)
    
    #print("%d. Nube: %d, Overall relative (Precision) Standard Deviation."%(contador,nodes))
    
    #UNCERTAINTY
    #df_uncer=df_uncer.append(uncertainty(nodes, df_window), ignore_index = True)
    df_uncer=uncertainty(nodes, df_window)
    
    #print("%d. Nube: %d, Overall relative Uncertainty BS, "%(contador,nodes))
    
    #ACCURACY
    #df_accur=df_accur.append(accuracy(nodes, df_window,Distances,SS), ignore_index = True)
    df_accur=accuracy(nodes, df_window,Distances,SS)
                             
    #print("%d. Nube: %d, Accuracy," %(contador, nodes))
    
    
    #COMPLETENESS
    #df_comple=df_comple.append(completeness(nodes, df_window), ignore_index = True)
    df_comple=completeness(nodes, df_window, start_time, end_time)
    #print("%d. Nube: %d, Completeness." %(contador,nodes))

    #COMPARABILITY / Concordance
    #df_conco=df_conco.append(concordance(nodes,df_window,Distances,SS),ignore_index = True)
    df_conco=concordance(nodes,df_window,Distances,SS)
                             
    #print("%d. Nube: %d, Overall concordance, "%(contador,nodes))
    print(nodes)
    return df_preci, df_uncer, df_accur, df_comple, df_conco
    

