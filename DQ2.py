#arguments=[nodes=10, CC=CC, SS=CC, Distances=Distances, start_time=start_time, end_time=end_time]
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import uniform
from scipy.stats import norm
from scipy import stats
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


#hola

#FUNCTION TO READ THE PATH WITH A DIALOG BOX
def get_path(wildcard, title):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, title, wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

#FUNCTION TO CLEAN AND SORT THE READ DATA.
def clean_sort_data(df_CC, df_SS):
    
    #ClEAN AND SORT THE CITIZEN SCIENCE DATASET
    #Remove outliers that are out of range, from documentation both nova and df range of measurements are [0,999], [-40,70], [1,100]
    #For PM2.5, Temperature and Relative Humidity Respectively.
    df_CC=df_CC.copy()
    
    df_CC.loc[df_CC["pm25_nova"]>999,"pm25_nova"]=np.nan
    df_CC.loc[df_CC["pm25_nova"]<0,"pm25_nova"]=np.nan
    
    df_CC.loc[df_CC["pm25_df"]>999,"pm25_df"]=np.nan
    df_CC.loc[df_CC["pm25_df"]<0,"pm25_df"]=np.nan
    
    df_CC.loc[df_CC["temperatura"]>70,"temperatura"]=np.nan
    df_CC.loc[df_CC["temperatura"]<-40,"temperatura"]=np.nan
    
    df_CC.loc[df_CC["humedad_relativa"]>100,"humedad_relativa"]=np.nan
    df_CC.loc[df_CC["humedad_relativa"]<1,"humedad_relativa"]=np.nan

    
    #Remove data above the whiskers of the boxplot: i.e. anomaly data
    Q1 = df_CC['pm25_df'].quantile(0.25)
    Q3 = df_CC['pm25_df'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_CC.loc[df_CC["pm25_df"]>=Q3 + 1.5 *IQR,"pm25_df"]=np.nan
    
    Q1 = df_CC['pm25_nova'].quantile(0.25)
    Q3 = df_CC['pm25_nova'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_CC.loc[df_CC["pm25_nova"]>=Q3 + 1.5 *IQR,"pm25_nova"]=np.nan
    
    Q1 = df_CC['temperatura'].quantile(0.25)
    Q3 = df_CC['temperatura'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_CC.loc[df_CC["temperatura"]>=Q3 + 1.5 *IQR,"temperatura"]=np.nan
    
    Q1 = df_CC['humedad_relativa'].quantile(0.25)
    Q3 = df_CC['humedad_relativa'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_CC.loc[df_CC["humedad_relativa"]>=Q3 + 1.5 *IQR,"humedad_relativa"]=np.nan
    
    #CREATE A DICTINARY CONTAINING THE DATASETS PER NODE
    grouped=df_CC.groupby(df_CC.codigoSerial)
    CC={}
    print("Citizen Scientist: ", sorted(list(df_CC.codigoSerial.unique())))
    for i in df_CC.codigoSerial.unique():
        CC[i] = grouped.get_group(i).sort_values(by=['fechaHora'],ignore_index=True)
    
    #ClEAN AND SORT THE SIATA STATIONS DATASET
    #Remove outliers that are out of range, from documentation both nova and df range of measurements are [0,999]
    df_SS=df_SS.copy()
    df_SS.loc[df_SS["pm25"]>999,"pm25"]=np.nan
    df_SS.loc[df_SS["pm25"]<0,"pm25"]=np.nan

    
    #Remove data above the whiskers of the boxplot: i.e. anomaly data
    Q1 = df_SS['pm25'].quantile(0.25)
    Q3 = df_SS['pm25'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    df_SS.loc[df_SS["pm25"]>=Q3 + 1.5 *IQR,"pm25"]=np.nan
    
    #CREATE A DICTINARY CONTAINING THE DATASETS PER SIATA STATION
    grouped=df_SS.groupby(df_SS.codigoSerial)
    SS={}
    print("Siata Stations: ", list(df_SS.codigoSerial.unique()))
    for j in df_SS.codigoSerial.unique():
        SS[j] = grouped.get_group(j).sort_values(by=['Fecha_Hora'],ignore_index=True)
    
    del grouped
    return CC, SS




#Precision
def precision(window):

    prec_df=max(0,1-(window['pm25_df'].std()/window['pm25_df'].mean()))# 1-Coefficient of Variation (std/mean)
    prec_nova=max(0,1-(window['pm25_nova'].std()/window['pm25_nova'].mean()))# 1-Coefficient of Variation (std/mean)
    prec_dict_time={"precision_df_time":prec_df,"precision_nova_time":prec_nova}
    return prec_dict_time


#Uncertainty
def uncertainty(window):
    uncert=max(0,1-np.sqrt((window.pm25_df-window.pm25_nova).pow(2).mean()/2)/((window.pm25_df+window.pm25_nova).mean()/2))
    uncer_dict_time={"uncertainty_time":uncert}
    return uncer_dict_time   


#ACCURACY
def accuracy(node,hour, window, Distances, SS):
    
    pm25_df_ave=window['pm25_df'].mean()
    pm25_nova_ave=window['pm25_nova'].mean()
    #print(pm25_df_ave, pm25_nova_ave)
    Closest_Station=Distances.codigoSerial_ES.loc[node]    
    Closest_Station2=Distances.codigoSerial_ES.loc[node] 
    #print(pm25_df_ave,pm25_nova_ave,Closest_Station,Closest_Station2)
    #print(Closest_Station in SS.keys())
    #print(hour in SS[Closest_Station].Fecha_Hora.values, hour)
   # print(SS[Closest_Station].Fecha_Hora)
    
    if (Closest_Station in SS.keys()) and (hour in SS[Closest_Station].Fecha_Hora.values):
        
        #print(SS[Closest_Station])
        #df_window.loc[df_window["fechaHora"]==ts,'pm25_nova_ave']
        v=SS[Closest_Station].loc[SS[Closest_Station].Fecha_Hora==hour,"pm25"].values[0]
        #print()
        #print(pm25_df_ave)
        accur_df=  max(0,1-abs(pm25_df_ave-v)/v)
        accur_nova= max(0,1-abs(pm25_nova_ave-v)/v)
    
    elif (Closest_Station2 in SS.keys()) and  (hour in SS[Closest_Station2].Fecha_Hora.values):
        
        v=SS[Closest_Station2].loc[SS[Closest_Station2].Fecha_Hora==hour,"pm25"].values[0]
        #v=SS[Closest_Station2].loc[hour,"pm25"]
        accur_df=  max(0,1-abs(pm25_df_ave-v)/v)
        accur_nova= max(0,1-abs(pm25_nova_ave-v)/v)
    else:
        accur_df=np.nan
        accur_nova=np.nan
    accu_dict_time={'accuracy_df_time':accur_df, 'accuracy_nova_time':accur_nova}
    return accu_dict_time


#CONCORDANCE
def concordance(node, hour, window, Distances, SS):
    Closest_Station=Distances.codigoSerial_ES.loc[node]
    Closest_Station2=Distances.codigoSerial_ES.loc[node]
    
    if (Closest_Station in SS.keys()) and (hour in SS[Closest_Station].Fecha_Hora.values):

        v=SS[Closest_Station].loc[SS[Closest_Station].Fecha_Hora==hour,"pm25"].values[0]
            
    elif (Closest_Station2 in SS.keys()) and (hour in SS[Closest_Station2].Fecha_Hora.values):
        
        v=SS[Closest_Station2].loc[SS[Closest_Station2].Fecha_Hora==hour,"pm25"].values[0]

    else:
       
        v=np.nan
        
    window=window.copy()
    window.loc[:,"v_pm25"]=v
    vm_df=window.pm25_df.mean()
    vm_nova=window.pm25_nova.mean()
    pair=[vm_df,vm_nova,v]
    corr_df =   window.loc[:,["pm25_df","pm25_nova","v_pm25","temperatura","humedad_relativa"]].corr().iloc[0].abs()
    corr_nova = window.loc[:,["pm25_df","pm25_nova","v_pm25","temperatura","humedad_relativa"]].corr().iloc[1].abs()
    concordance_df_nova=corr_df.pm25_nova
    concordance_df_siata=corr_df.v_pm25
    concordance_df_hum=corr_df.humedad_relativa
    concordance_df_temp=corr_df.temperatura
    concordance_nova_siata=corr_nova.v_pm25
    concordance_nova_hum=corr_nova.humedad_relativa
    concordance_nova_temp=corr_nova.temperatura
       
    conco_dict={"concordance_df_nova_time":concordance_df_nova,
                "concordance_df_siata_time":concordance_df_siata,
                "concordance_df_hum_time":concordance_df_hum,
                "concordance_df_temp_time":concordance_df_temp,
                "concordance_nova_siata_time":concordance_nova_siata,
                "concordance_nova_hum_time":concordance_nova_hum,
                "concordance_nova_temp_time":concordance_nova_temp,
                "vm_df":pair[0],
                "vm_nova":pair[1],
                "v":pair[2]}
    return conco_dict

#COMPLETENESS
def completeness(node, window,start_time, end_time):
    
    if window.fechaHora.min()==start_time and window.fechaHora.max()==end_time:
        ref_date_range = pd.DataFrame(pd.date_range(start_time,end_time, freq='1Min'),columns=["ref_fechaHora"])
    
    elif window.fechaHora.min()==start_time:
        ref_date_range = pd.DataFrame(pd.date_range(start_time, pd.Timestamp(start_time).ceil('60min')-timedelta(minutes=1), freq='1Min'),columns=["ref_fechaHora"])
        
    elif window.fechaHora.max()==end_time:
        ref_date_range = pd.DataFrame(pd.date_range(pd.Timestamp(end_time).floor('60min')-timedelta(minutes=1),pd.Timestamp(end_time), freq='1Min'),columns=["ref_fechaHora"])
    
    else:
        ref_date_range = pd.DataFrame(pd.date_range(window.fechaHora.min().floor('60min'),window.fechaHora.min().floor('60min')+timedelta(minutes=59), freq='1Min'),columns=["ref_fechaHora"])
        
    #Check for any missing date
    missing_dates = ref_date_range.loc[~ref_date_range.ref_fechaHora.isin(window.fechaHora),"ref_fechaHora"]
    
    #Add missing date rows
    for missing in missing_dates:
        window=window.append({"codigoSerial":node,"fechaHora":missing}, ignore_index = True)
    #Check for missing data
    missing_data_df=np.count_nonzero(np.isnan(window['pm25_df']))
    missing_data_nova=np.count_nonzero(np.isnan(window['pm25_nova']))
    comp_df=(1-missing_data_df/np.size(window.pm25_df))
    comp_nova=(1-missing_data_nova/np.size(window.pm25_nova))
    comp_dict_time={'completeness_df_time':comp_df, 'completeness_nova_time':comp_nova}        
    return comp_dict_time

#REDUNDANCY/DUPLICATES
def duplicates(window):
    repeated=len(window.index)-window['fechaHora'].nunique()
    duplic=1-repeated/len(window.index)
    dupli_dict_time={'duplicates_time':duplic} 
    return dupli_dict_time

#CONFIDENCE
def confidence(window, STD_df, STD_nova, p):
    
    alpha = 1-p/100
    z = stats.norm.ppf(1-alpha/2)#2-sided test
    
    n=np.count_nonzero(~np.isnan(window['pm25_df']))
    confi_df  = max(0,1 - (z*STD_df/np.sqrt(n))/window.pm25_df.mean())
    
    n=np.count_nonzero(~np.isnan(window['pm25_nova']))
    confi_nova= max(0,1 - (z*STD_df/np.sqrt(n))/window.pm25_nova.mean())
       
    confi_dict_time={'confi_df_time':confi_df,'confi_nova_time':confi_nova}
    return confi_dict_time

def eval_dq(arguments):
    nodes=arguments[0]
    CC=arguments[1]
    SS=arguments[2]
    Distances=arguments[3]
    start_time=arguments[4]
    end_time=arguments[5]
    p=arguments[6]
    print("no se imprime")
    #1. For each citizen science (CC) node, get the groups (HOURLY GROUPS).
    #node_dataset=CC[nodes]
    node_dataset=CC[nodes][(CC[nodes]['fechaHora'] >= start_time) & (CC[nodes]['fechaHora'] <= end_time)]
    STD_df=node_dataset.pm25_df.std()
    STD_nova=node_dataset.pm25_nova.std()
    #print(node_dataset)
    #times = pd.to_datetime(node_dataset.fechaHora)
    hourly_groups=node_dataset.groupby([node_dataset.fechaHora.dt.floor('60min')])#Para agrupar por cada hora
    #hourly_groups.groups.keys()# O grupos.groups para obtener las claves de cada grupo, es decir cada hora
    del node_dataset
    

    dim_time = pd.DataFrame(
        columns =["codigoSerial",
                  "fechaHora",
                  "precision_df_time",
                  "precision_nova_time",
                  "uncertainty_time",
                  "accuracy_df_time",
                  "accuracy_nova_time",
                  "completeness_df_time",
                  "completeness_nova_time",
                  
                  "concordance_df_nova_time",
                  
                  "concordance_df_siata_time",
                  "concordance_df_hum_time",
                  "concordance_df_temp_time",
                  
                  "concordance_nova_siata_time",
                  "concordance_nova_hum_time",
                  "concordance_nova_temp_time",
                  "vm_df",
                  "vm_nova",
                  "v",
                 
                  "duplicates_time",
                  
                  "confi_df_time",
                  "confi_nova_time"])
    
    #2. For each group (hour in a CC node data), calculate the Dimension's DQ. (The functions should be applied to each group instead)
    for hour in hourly_groups.groups.keys():
        window=hourly_groups.get_group(hour)
        
        
        preci_dict_time=precision(window)
        uncer_dict_time=uncertainty(window)
        accur_dict_time=accuracy(nodes, hour, window, Distances, SS)
        conco_dict_time=concordance(nodes, hour, window, Distances, SS)
        comp_dict_time=completeness(nodes, window,start_time, end_time)
        dupli_dict_time=duplicates(window)
        confi_dict_time=confidence(window, STD_df, STD_nova, p)
        
        DQ_dict_time={"codigoSerial":nodes,"fechaHora":pd.Timestamp(hour)}
        DQ_dict_time.update(preci_dict_time)
        DQ_dict_time.update(uncer_dict_time)
        DQ_dict_time.update(accur_dict_time)
        DQ_dict_time.update(conco_dict_time)
        DQ_dict_time.update(comp_dict_time)
        DQ_dict_time.update(dupli_dict_time)
        DQ_dict_time.update(confi_dict_time)
        
        
        #3. Save the result file in the form: Node, Group (hour), DQ_1, DQ_2, DQ3, ... , DQIndex  (This is new)
        dim_time=dim_time.append(DQ_dict_time, ignore_index = True)
    
    dim_time['fechaHora']= pd.to_datetime(dim_time['fechaHora'])
    daily_groups=dim_time.groupby([dim_time.fechaHora.dt.floor('1D')])#Para agrupar por cada día

    for day in daily_groups.groups.keys():
        day_window=daily_groups.get_group(day)
        indexes=(dim_time.fechaHora>=day.floor('1D')) & (dim_time.fechaHora<(day+timedelta(minutes=1)).ceil('1D'))
        dim_time.loc[indexes,"concordance_df_siata_time"]=abs(day_window.v.corr(day_window.vm_df))
        dim_time.loc[indexes,"concordance_nova_siata_time"]=abs(day_window.v.corr(day_window.vm_nova))

    
    
    
    col =        ["precision_df_time",
                  "precision_nova_time",
                  "uncertainty_time",
                  "accuracy_df_time",
                  "accuracy_nova_time",
                  "completeness_df_time",
                  "completeness_nova_time",
                  
                  "concordance_df_nova_time",
                  
                  "concordance_df_siata_time",
                  "concordance_df_hum_time",
                  "concordance_df_temp_time",
                  
                  "concordance_nova_siata_time",
                  "concordance_nova_hum_time",
                  "concordance_nova_temp_time",
                 
                  "duplicates_time",
                  
                  "confi_df_time",
                  "confi_nova_time"]
    
    #dim_node = pd.DataFrame(columns =["codigoSerial"])
    
    dim_node= dim_time[col].mean()
    dim_node.rename({'precision_df_time': 'precision_df_node', 
                     'precision_nova_time': 'precision_nova_node', 
                     'uncertainty_time': 'uncertainty_node', 
                     'accuracy_df_time': 'accuracy_df_node', 
                     'accuracy_nova_time': 'accuracy_nova_node', 
                     'completeness_df_time': 'completeness_df_node', 
                     'completeness_nova_time': 'completeness_nova_node', 
                     'concordance_df_nova_time': 'concordance_df_nova_node', #
                     'concordance_df_siata_time': 'concordance_df_siata_node', #
                     'concordance_df_hum_time': 'concordance_df_hum_node', 
                     'concordance_df_temp_time': 'concordance_df_temp_node', 
                     'concordance_nova_siata_time': 'concordance_nova_siata_node', #
                     'concordance_nova_hum_time': 'concordance_nova_hum_node', 
                     'concordance_nova_temp_time': 'concordance_nova_temp_node', 
                     'duplicates_time': 'duplicates_node', 
                     'confi_df_time': 'confi_df_node', 
                     'confi_nova_time': 'confi_nova_node'        }, axis=1, inplace=True)
    
    dim_node["codigoSerial"]=nodes

        
    return [dim_time, dim_node]
