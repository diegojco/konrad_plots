"""A module that helps to plot the results of `konrad`"""

import numpy as np
from netCDF4 import Dataset as ncload
import matplotlib.pyplot as plt
from typhon.plots import formatter

# Constants

## String constants

### Units

t_u       = r"\mathrm{step}"
p_u       = r"\mathrm{hPa}"
z_u       = r"\mathrm{km}"
T_u       = r"\mathrm{K}"
dTdz_u    = r"\mathrm{K}\,\mathrm{km}^{-1}"
I_u       = r"\mathrm{W}\,\mathrm{m}^{-2}"
sigma_u   = r"1\times 10^{-6}"

### Quantity symbols

t_s       = r"t"
p_s       = r"p"
T_s       = r"T"
dTdz_s    = r"\Gamma"
I_lwu_s   = r"I_{\mathrm{lw,u}}"
I_swu_s   = r"I_{\mathrm{sw,u}}"
I_lwd_s   = r"I_{\mathrm{lw,d}}"
I_swd_s   = r"I_{\mathrm{sw,d}}"
I_lw_s    = r"I_{\mathrm{lw,net}}"
I_sw_s    = r"I_{\mathrm{sw,net}}"
I_s       = r"I_{\mathrm{net}}"
O3_s      = r"\sigma_{\mathrm{O}_{3}}"
ct_s      = r"h_{\mathrm{conv.\,top}}"
cp_s      = r"h_{\mathrm{cold\,point}}"
cpT_s     = r"T_{\mathrm{cold\,point}}"

### Difference symbols

Dt_s      = r"\Delta "+t_s
Dp_s      = r"\Delta "+p_s
DT_s      = r"\Delta "+T_s
DdTdz_s   = r"\Delta "+dTdz_s
DI_lwu_s  = r"\Delta "+I_lwu_s
DI_swu_s  = r"\Delta "+I_swu_s
DI_lwd_s  = r"\Delta "+I_lwd_s
DI_swd_s  = r"\Delta "+I_swd_s
DI_lw_s   = r"\Delta "+I_lw_s
DI_sw_s   = r"\Delta "+I_sw_s
DI_s      = r"\Delta "+I_s
DO3_s     = r"\Delta "+O3_s
Dct_s     = r"\Delta "+ct_s
Dcp_s     = r"\Delta "+cp_s
DcpT_s    = r"\Delta "+cpT_s

### Symbols with units for axes labelling

t_a       = r"$"+t_s+r"/"+t_u+r"$"
p_a       = r"$"+p_s+r"/"+p_u+r"$"
T_a       = r"$"+T_s+r"/"+T_u+r"$"
dTdz_a    = r"$"+dTdz_s+r"/"+dTdz_u+r"$"
I_lwu_a   = r"$"+I_lwu_s+r"/"+I_u+r"$"
I_swu_a   = r"$"+I_swu_s+r"/"+I_u+r"$"
I_lwd_a   = r"$"+I_lwd_s+r"/"+I_u+r"$"
I_swd_a   = r"$"+I_swd_s+r"/"+I_u+r"$"
I_lw_a    = r"$"+I_lw_s+r"/"+I_u+r"$"
I_sw_a    = r"$"+I_sw_s+r"/"+I_u+r"$"
I_a       = r"$"+I_s+r"/"+I_u+r"$"
O3_a      = r"$"+O3_s+r"/"+sigma_u+r"$"
ct_a      = r"$"+ct_s+r"/"+z_u+r"$"
cp_a      = r"$"+cp_s+r"/"+z_u+r"$"
cpT_a      = r"$"+cpT_s+r"/"+T_u+r"$"

Dt_a      = r"$"+Dt_s+r"/"+t_u+r"$"
Dp_a      = r"$"+Dp_s+r"/"+p_u+r"$"
DT_a      = r"$"+DT_s+r"/"+T_u+r"$"
DdTdz_a   = r"$"+DdTdz_s+r"/"+dTdz_u+r"$"
DI_lwu_a  = r"$"+DI_lwu_s+r"/"+I_u+r"$"
DI_swu_a  = r"$"+DI_swu_s+r"/"+I_u+r"$"
DI_lwd_a  = r"$"+DI_lwd_s+r"/"+I_u+r"$"
DI_swd_a  = r"$"+DI_swd_s+r"/"+I_u+r"$"
DI_lw_a   = r"$"+DI_lw_s+r"/"+I_u+r"$"
DI_sw_a   = r"$"+DI_sw_s+r"/"+I_u+r"$"
DI_a      = r"$"+DI_s+r"/"+I_u+r"$"
DO3_a     = r"$"+DO3_s+r"/"+sigma_u+r"$"
Dct_a     = r"$"+Dct_s+r"/"+z_u+r"$"
Dcp_a     = r"$"+Dcp_s+r"/"+z_u+r"$"
DcpT_a    = r"$"+DcpT_s+r"/"+T_u+r"$"

## Profiles that can be plotted

### Ingredients: Necessary fields

ext_def = {
 "temp"  : [("atmosphere","T")],
 "lapse" : [("atmosphere","T"),("atmosphere","z")],
 "rlwd"  : [("radiation","lw_flxd")],
 "rswd"  : [("radiation","sw_flxd")],
 "rlwu"  : [("radiation","lw_flxu")],
 "rswu"  : [("radiation","sw_flxu")],
 "rlw"   : [("radiation","lw_flxd"),("radiation","lw_flxu")],
 "rsw"   : [("radiation","sw_flxd"),("radiation","sw_flxu")],
 "rnet"  : [("radiation","lw_flxd"),("radiation","lw_flxu"),
            ("radiation","sw_flxd"),("radiation","sw_flxu")],
 "O3"    : [("atmosphere","O3")],
 "ct"    : [("convection","convective_top_height")],
 "cp"    : [("atmosphere","T"),("atmosphere","z")],
 "cpT"   : [("atmosphere","T"),("atmosphere","z")],
}

### Time indexes

tim_ser = {
 "temp"  : (-1,-1),
 "lapse" : (-1,-1),
 "rlwd"  : (-1,-1),
 "rswd"  : (-1,-1),
 "rlwu"  : (-1,-1),
 "rswu"  : (-1,-1),
 "rlw"   : (-1,-1),
 "rsw"   : (-1,-1),
 "rnet"  : (-1,-1),
 "O3"    : (-1,-1),
 "ct"    : (0,-1),
 "cp"    : (0,-1),
 "cpT"   : (0,-1),
}

### Recipes: What to do with the collected fields

#### Helper functions

def identity(datae,factor=1):
 """Identity function
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 
 Returns
 -------
 
 numpy.ndarray
            Returns the first element of the list. Thought to process a list of
            one field.
 
 """
 return datae[0]*factor

def additive_inverse(datae):
 """Returns the additive inverse
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 
 Returns
 -------
 
 numpy.ndarray
            Returns the additive inverse of the first element of the list.
            Thought to process a list of one field.
 
 """
 return -datae[0]

def subtract(datae):
 """Subtracts two fields
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 
 Returns
 -------
 
 numpy.ndarray
            Subtracts the second field to the first field in `datae`. Thought to
            process a list of two fields.
 
 """
 return datae[0] - datae[1]

def add_differences(datae):
 """Addition of two differences of fields
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 
 Returns
 -------
 
 numpy.ndarray
            Subtracts the second field to the first field in `datae`. It does
            the same with the third and four fields. Thought to process a list
            of four fields.
            
 """
 
 return subtract(datae[0:2])+subtract(datae[2:])

def gradient(datae,factor=1000):
 """Derivative
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 factor   : scalar, optional
            Conversion factor. The default is 1000, for converting from meters
            to kilometers: the function is mainly thought as for calculating
            vertical lapse rates of atmospheric quantities 
 
 Returns
 -------
 
 numpy.ndarray
            Calculates the gradient of `datae[0]` with respect to `datae[1]`.
            Thought to process a list of two fields.
            
 """
 
 return np.gradient(datae[0],datae[1])*factor

def cold_point(datae,factor=1000):
 """Obtains a cold point height time series
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 factor   : scalar, optional
            Conversion factor. The default is 1000, for converting from meters
            to kilometers. 
 
 Returns
 -------
 
 cp       : numpy.ndarray
            Cold point time series
            
 """
 cp = np.array([ np.gradient(datae[0][i],datae[1][i])*factor 
                 for i in range(len(datae[0])) ])
 cp = np.array([ datae[1][i,np.where(cp[i] >= 0)[0][0]]
                 for i in range(len(datae[0])) ])/factor
 return cp

def cold_point_temperature(datae,factor=1000):
 """Obtains a cold point temperature time series
 
 Parameters
 ----------
 
 datae    : list
            List of `np.ndarray` with the fields on which the function operates.
 factor   : scalar, optional
            Conversion factor. The default is 1000, for converting from meters
            to kilometers. 
 
 Returns
 -------
 
 cpT      : numpy.ndarray
            Cold point temperature time series
            
 """
 cpT = np.array([ np.gradient(datae[0][i],datae[1][i])*factor 
                  for i in range(len(datae[0])) ])
 cpT = np.array([ datae[0][i,np.where(cpT[i] >= 0)[0][0]]
                  for i in range(len(datae[0])) ])
 return cpT

#### Dictionary of recipes

clc_def = {
 "temp"  : identity,
 "lapse" : gradient,
 "rlwd"  : identity,
 "rswd"  : identity,
 "rlwu"  : additive_inverse,
 "rswu"  : additive_inverse,
 "rlw"   : subtract,
 "rsw"   : subtract,
 "rnet"  : add_differences,
 "O3"    : lambda datae: identity(datae,factor=1e6),
 "ct"    : lambda datae: identity(datae,factor=1e-3),
 "cp"    : cold_point,
 "cpT"   : cold_point_temperature
}

### Labelling: Profile name

variable_label = {
 "temp"  : T_a,
 "lapse" : dTdz_a,
 "rlwd"  : I_lwd_a,
 "rswd"  : I_swd_a,
 "rlwu"  : I_lwu_a,
 "rswu"  : I_swu_a,
 "rlw"   : I_lw_a,
 "rsw"   : I_sw_a,
 "rnet"  : I_a,
 "O3"    : O3_a,
 "ct"    : ct_a,
 "cp"    : cp_a,
 "cpT"   : cpT_a,
}

Dvariable_label = {
 "temp"  : DT_a,
 "lapse" : DdTdz_a,
 "rlwd"  : DI_lwd_a,
 "rswd"  : DI_swd_a,
 "rlwu"  : DI_lwu_a,
 "rswu"  : DI_swu_a,
 "rlw"   : DI_lw_a,
 "rsw"   : DI_sw_a,
 "rnet"  : DI_a,
 "O3"    : DO3_a,
 "ct"    : Dct_a,
 "cp"    : Dcp_a,
 "cpT"   : DcpT_a,
}

# Extraction functions

def extract(group,variable,file,
            idxi=0,
            idxf=-1
           ):
 """Extracts a time series from `konrad` output.
 
 Parameters
 ----------
 
 group    : str
            Group name in the output file.
 variable : str
            Variable name in the output file.
 file     : str
            Path to the output file.
 idxi     : int, optional
            The initial index in the time dimension (the first is the default).
 idxf     : int, optional
            The final index in the time dimension (the last is the default).
 
 Returns
 -------
 
 numpy.ndarray
            The time series of the `variable`.
            
 """
 with ncload(file,mode="r") as f:
  if idxi == idxf:
   temp = f.groups[group][variable][idxi]
  else:
   if idxf == -1:
    temp = f.groups[group][variable][idxi:]
   else:
    temp = f.groups[group][variable][idxi:idxf]
  temp = np.array(temp,dtype="float64")
  return temp

def extract_plev(file,
                 plevtype="plev"
                ):
 """Extracts the pressure levels from `konrad` output.
 
 Parameters
 ----------
 
 file     : str
            Path to the output file.
 plevtype : {"plev", "phlev"}, optional
            The pressure level type: pressure full levels (`"plev"`) or pressure
            half levels (`"phlev"`). The default value is `"plev"`.
 
 Returns
 -------
 
 numpy.ndarray
            The list of pressure levels.
            
 """
 with ncload(file,mode="r") as f:
  temp = f.variables[plevtype][:]
  temp = np.array(temp,dtype="float64")
  return temp

def extract_time(file,
                 idxi=0,
                 idxf=-1
                ):
 """Extracts time values from `konrad` output.
 
 Parameters
 ----------
 
 file : str
        Path to the output file.
 idxi : int, optional
        The initial index in the time dimension (the first is the default).
 idxf : int, optional
        The final index in the time dimension (the last is the default).
 
 Returns
 -------
 
 numpy.ndarray
        The list of times.
            
 """
 with ncload(file,mode="r") as f:
  if idxi == idxf:
   temp = f.variables["time"][idxi]
  else:
   if idxf == -1:
    temp = f.variables["time"][idxi:]
   else:
    temp = f.variables["time"][idxi:idxf]
  temp = np.array(temp,dtype="float64")
  return temp

def get_from_konrad(variable,exps):
 """Extracts a variable from the output files of several `konrad` experiments.
 
 Parameters
 ----------
 
 variable : str
            Variable name in the output file.
 exps     : list
            List of strings describing the paths to the output files.
 
 Returns
 -------
 
 p,t      : numpy.ndarray
            Pressure levels or times corresponding to the variable.
 t_min    : scalar
            The minimum of times, if a time series.
 t_max    : scalar
            The maximum of times, if a time series.
 data     : dict
            The requested variable for all the experiments in `exps`.
 minimum  : scalar
            The minimum of all experiments.
 maximum  : scalar
            The maximum of all experiments.
            
 """
 
 idx = tim_ser[variable]                             # Select time indexes
 
 if variable in ["temp","lapse","O3"]:               # Select pressure levels
  p = extract_plev(exps[0])
 else:
  p = extract_plev(exps[0],plevtype="phlev")
  
 if variable in ["ct","cp","cpT"]:                   # Select times (for ts)
  t = [ extract_time(e,idxi=idx[0],idxf=idx[1]) for e in exps ]
  t = dict(zip(exps,t))
  
  t_min = np.array([ t[e].min() for e in t ]).min()
  t_max = np.array([ t[e].max() for e in t ]).max()
 
 data = [ [ extract(rule[0],rule[1],e,idxi=idx[0],idxf=idx[1]) # Extract data
            for rule in ext_def[variable] ]
          for e in exps ]
 data = [ clc_def[variable](e) for e in data ]
 data = dict(zip(exps,data))
 
 minimum = np.array([ data[e].min() for e in data ]).min() # Get extrema
 maximum = np.array([ data[e].max() for e in data ]).max()
 
 if variable in ["ct","cp","cpT"]:
  return (t,t_min,t_max,data,minimum,maximum)
 else:
  return (p,data,minimum,maximum)

def get_from_konrad_ref(variable,exps,exp_ref):
 """Extracts a variable from the output files of several `konrad` experiments
 and calculates the differences from a reference experiment.
 
 Parameters
 ----------
 
 variable     : str
                Variable name in the output file.
 exps         : list
                List of strings describing the paths to the output files.
 exp_ref      : str
                Path to the output file of the reference experiment.
 
 Returns
 -------
 
 p,t          : numpy.ndarray
                Pressure levels or times corresponding to the variable.
 t_min        : scalar
                The minimum of times, if a time series.
 t_max        : scalar
                The maximum of times, if a time series.
 data_diff    : dict
                The variable relative differences for all experiments in `exps`.
 minimum_diff : scalar
                The minimum of all experiments.
 maximum_diff : scalar
                The maximum of all experiments.
 data_ref     : numpy.ndarray
                The variable values for `exp_ref`.
 minimum_ref  : scalar
                The reference minimum.
 maximum_ref  : scalar
                The reference maximum.
                
 """
 
 idx = tim_ser[variable]                             # Select time indexes
 
 if variable in ["temp","lapse","O3"]:               # Select pressure levels
  p = extract_plev(exps[0])
 else:
  p = extract_plev(exps[0],plevtype="phlev")
  
 if variable in ["ct","cp","cpT"]:                   # Select times (for ts)
  t_ref = extract_time(exp_ref,idxi=idx[0],idxf=idx[1])
  
  t = [ extract_time(e,idxi=idx[0],idxf=idx[1]) for e in exps ]
  t = dict(zip(exps,t))
  
  t[exp_ref] = t_ref
  
  t_min = np.array([ t[e].min() for e in t ]).min()
  t_max = np.array([ t[e].max() for e in t ]).max()
 
 data_ref = [ extract(rule[0],rule[1],exp_ref,idxi=idx[0],idxf=idx[1])
              for rule in ext_def[variable] ]
 data_ref = clc_def[variable](data_ref)              # Select reference data
 
 data = [ [ extract(rule[0],rule[1],e,idxi=idx[0],idxf=idx[1])
            for rule in ext_def[variable] ]
          for e in exps ]
 data = [ clc_def[variable](e) for e in data ]       # Select data
 if variable in ["ct","cp","cpT"]:                   # Obtain the differences
  data = [ e - data_ref[-1] for e in data ]
 else:
  data = [ e - data_ref for e in data ]
 data = dict(zip(exps,data))
 
 minimum_diff = np.array([ data[e].min() for e in data ]).min() # Get extrema
 maximum_diff = np.array([ data[e].max() for e in data ]).max()
 minimum_ref = data_ref.min()
 maximum_ref = data_ref.max()
 
 data[exp_ref] = data_ref                            # Pack ref. data together
 
 if variable in ["ct","cp","cpT"]:
  return (t,t_min,t_max,
          data,minimum_diff,maximum_diff,
          minimum_ref,maximum_ref
         )
 else:
  return (p,
          data,minimum_diff,maximum_diff,
          minimum_ref,maximum_ref
         )

# Plotting functions

def plot_from_konrad(variable,exps,
                     same=False,
                     delta=10,
                     xrefline=None,
                     title=None,
                     labels=None,
                     styles=None
                    ):
 """Plots variable profiles from several `konrad` experiments.
 
 Parameters
 ----------
 
 variable       : str
                  Variable name in the output file.
 exps           : list
                  List of strings describing the paths to the output files.
 same           : bool, optional
                  If `True`, profiles are plotted in the same pair of axes.
                  Otherwise, there are as many pairs of axes as profiles
                  (`False` is the default).
 delta          : scalar, optional
                  Offset added to the extrema of the data.
 xrefline       : scalar or `None`, optional
                  If `same` is `True`, plots a vertical reference line at the
                  given scalar value. If `same` is `False`, plots vertical
                  reference lines at each axis. If `None`, plots no vertical
                  reference lines (`None` is the default).
 title          : str or `None`, optional
                  An overarching title for the figure (`None` is the default).
 labels, styles : dict or `None`, optional
                  Dictionaries containing profile labels and line styles.
                  Usually created by zipping two lists, e.g.
                  `labels = dict(zip(exps,labels))`. The `styles` dictionaries
                  are passed as keyword arguments to the `plt.plot` function.
                  If `None`, one gets default styles and/or labels.
 
 Returns
 -------
 
 fig            : matplotlib.Figure
                  Figure with the plots.
 axes           : matplotlib.Axes
                  Axes for the figure.
                  
 """
 
 (p,data,minimum,maximum) = get_from_konrad(variable,exps) # Gets the data
 n = len(exps)                                             # Experiment count
 
 width = 1                                                 # Width (one pair)
 height = 5                                                # Height
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(3*width,height))
  axes = fig.add_subplot(1,1,1)
  
  for e in data:                                           # Plotting
   ax = axes
   if labels == None:
    if styles == None:
     ax.plot(data[e],p)
    elif styles[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(data[e],p)
    elif styles[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**styles[e])
   else:
    if styles == None:
     ax.plot(data[e],p,label=labels[e])
    elif styles[e] == None:
     ax.plot(data[e],p,label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(data[e],p,**labels_and_styles)
  
  ax = axes
  ax.invert_yaxis()                                        # Settings for y-axis
  ax.set_yscale("log")
  formatter.set_yaxis_formatter(formatter.HectoPascalLogFormatter(),
                                ax=ax)
  ax.tick_params(axis='y',which='both',labelsize=6)
  ax.set_ylabel(p_a,fontsize=6)
  
  ax.set_xlim(minimum - delta,maximum + delta)             # Settings for x-axis
  ax.tick_params(axis='x',which='both',labeltop=False,labelsize=6)
  ax.set_xlabel(variable_label[variable],fontsize=6)
  
  ax.minorticks_off()                                      # Unsets minor ticks
  
  if xrefline != None:                                     # Reference line
   ax.axvline(x=xrefline,lw=0.7,color=(0,0,0))
  
  ax.legend(fontsize=5)                                    # Legend
 
 else:
  
  fig = plt.figure(dpi=300,figsize=(n*width,height))
  axes = { exps[i-1] : fig.add_subplot(1,n,i) for i in range(1,n+1) }
  
  for e in axes:                                           # Plotting
   ax = axes[e]
   if labels == None:
    if styles == None:
     ax.plot(data[e],p)
    elif styles[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(data[e],p)
    elif styles[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**styles[e])
   else:
    if styles == None:
     ax.plot(data[e],p,label=labels[e])
    elif styles[e] == None:
     ax.plot(data[e],p,label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(data[e],p,**labels_and_styles)
   
   ax.invert_yaxis()                                       # Settings for y-axes
   ax.set_yscale("log")
   if e != exps[0]:
    ax.tick_params(axis='y',which='both',labelleft=False)
   if e == exps[0]:
    formatter.set_yaxis_formatter(formatter.HectoPascalLogFormatter(),
                                  ax=ax)
    ax.tick_params(axis='y',which='both',labelsize=6)
    ax.set_ylabel(p_a,fontsize=6)
   
   ax.set_xlim(minimum - delta,maximum + delta)            # Settings for x-axes
   ax.tick_params(axis='x',which='both',labeltop=False,labelsize=6)
   ax.set_xlabel(variable_label[variable],fontsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if xrefline != None:                                    # Reference line
    ax.axvline(x=xrefline,lw=0.7,color=(0,0,0))
    
   ax.set_title(labels[e],fontsize=5)                      # Sets axes titles
   
 if title != None:                                         # Title settings
  fig.suptitle(title,fontsize=5,y=0.95)
 
 fig.set_tight_layout("tight")                             # Other settings
 
 return (fig,axes)

def plot_from_konrad_ts1(variable,exps,
                         same=False,
                         delta=0.5,
                         yrefline=None,
                         title=None,
                         labels=None,
                         styles=None
                        ):
 """Plots a time series of a 1d variable from several `konrad` experiments.
 
 Parameters
 ----------
 
 variable       : str
                  Variable name in the output file.
 exps           : list
                  List of strings describing the paths to the output files.
 same           : bool, optional
                  If `True`, profiles are plotted in the same pair of axes.
                  Otherwise, there are as many pairs of axes as profiles
                  (`False` is the default).
 delta          : scalar, optional
                  Offset added to the extrema of the data.
 yrefline       : scalar or `None`, optional
                  If `same` is `True`, plots a horizontal reference line at the
                  given scalar value. If `same` is `False`, plots horizontal
                  reference lines at each axis. If `None`, plots no horizontal
                  reference lines (`None` is the default).
 title          : str or `None`, optional
                  An overarching title for the figure (`None` is the default).
 labels, styles : dict or `None`, optional
                  Dictionaries containing profile labels and line styles.
                  Usually created by zipping two lists, e.g.
                  `labels = dict(zip(exps,labels))`. The `styles` dictionaries
                  are passed as keyword arguments to the `plt.plot` function.
                  If `None`, one gets default styles and/or labels.
 
 Returns
 -------
 
 fig            : matplotlib.Figure
                  Figure with the plots.
 axes           : matplotlib.Axes
                  Axes for the figure.
                  
 """
 (times,time_min,time_max,
  data,minimum,maximum) = get_from_konrad(variable,exps)   # Gets the data 
 n = len(exps)                                             # Experiment count
 
 width = 5                                                 # Width
 height = 1                                                # Height (one pair)
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(width,3*height))
  axes = fig.add_subplot(1,1,1)
  
  for e in data:                                           # Plotting
   ax = axes
   if labels == None:
    if styles == None:
     ax.plot(times[e],data[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(times[e],data[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**styles[e])
   else:
    if styles == None:
     ax.plot(times[e],data[e],label=labels[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e],label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(times[e],data[e],**labels_and_styles)
  
  ax = axes
  ax.set_xlim(time_min - 24,time_max + 24)
  ax.tick_params(axis='x',which='both',
                 labeltop=False,labelsize=6)               # Settings for x-axis
  ax.set_xlabel(t_a,fontsize=6)
  
  ax.set_ylim(minimum - delta,maximum + delta)             # Settings for y-axis
  ax.tick_params(axis='y',which='both',labelright=False,labelsize=6)
  ax.set_ylabel(variable_label[variable],fontsize=6)
  
  ax.minorticks_off()                                      # Unsets minor ticks
  
  if yrefline != None:                                     # Reference line
   ax.axhline(y=yrefline,lw=0.7,color=(0,0,0))
  
  ax.legend(fontsize=5)                                    # Legend
 
 else:
  
  fig = plt.figure(dpi=300,figsize=(width,n*height))
  axes = { exps[i-1] : fig.add_subplot(n,1,i) for i in range(1,n+1) }
  
  for e in axes:                                           # Plotting
   ax = axes[e]
   if labels == None:
    if styles == None:
     ax.plot(times[e],data[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(times[e],data[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**styles[e])
   else:
    if styles == None:
     ax.plot(times[e],data[e],label=labels[e])
    elif styles[e] == None:
     ax.plot(times[e],data[e],label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(times[e],data[e],**labels_and_styles)
   
   ax.set_xlim(time_min - 24,time_max + 24)
   if e != exps[-1]:                                       # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e == exps[-1]:
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel(t_a,fontsize=6)
   
   ax.set_ylim(minimum - delta,maximum + delta)            # Settings for y-axes
   ax.tick_params(axis='y',which='both',labelsize=6)
   ax.set_ylabel(variable_label[variable],fontsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if yrefline != None:                                    # Reference line
    ax.axhline(y=yrefline,lw=0.7,color=(0,0,0))
    
   ax.set_title(labels[e],fontsize=5)                      # Sets axes titles
   
 if title != None:                                         # Title settings
  fig.suptitle(title,fontsize=5,y=0.95)
 
 fig.set_tight_layout("tight")                             # Other settings
 
 return (fig,axes)

def plot_from_konrad_diff(variable,exps,exp_ref,
                          same=False,
                          delta=10,
                          delta_ref=5,
                          xrefline=0,
                          xrefline_ref=None,
                          title=None,
                          labels=None,
                          styles=None,
                          label_ref=None,
                          style_ref=None
                         ):
 """Plots variable profiles from several `konrad` experiments in relation to a
 given reference experiment.
 
 Parameters
 ----------
 
 variable       : str
                  Variable name in the output file.
 exps           : list
                  List of strings describing the paths to the output files.
 exp_ref        : str
                  String describing the path to the output file for the
                  reference experiment.
 same           : bool, optional
                  If `True`, profiles are plotted in the same pair of axes.
                  Otherwise, there are as many pairs of axes as profiles
                  (`False` is the default). Reference profile has always its own
                  pair of axes.
 delta          : scalar, optional
                  Offset added to the extrema of the data.
 delta_ref      : scalar, optional
                  Offset added to the extrema of the reference data.
 xrefline       : scalar, optional
                  If `same` is `True`, plots a vertical reference line at the
                  given scalar value. If `same` is `False`, plots vertical
                  reference lines at each axis. By default the lines are zero
                  lines.
 xrefline_ref   : scalar or `None` optional
                  Plots a vertical reference line at the given scalar value. If
                  `None`, no reference line is plotted: this is the default.
 title          : str or `None`, optional
                  An overarching title for the figure (`None` is the default).
 labels, styles : dict or `None`, optional
                  Dictionaries containing profile labels and line styles.
                  Usually created by zipping two lists, e.g.
                  `labels = dict(zip(exps,labels))`. The `styles` dictionaries
                  are passed as keyword arguments to the `plt.plot` function. If
                  `None`, one gets default styles and/or labels.
 label_ref      : str or `None`, optional
                  String containing reference profile label.
 style_ref      : dict or `None`, optional
                  Dictionary containing reference profile line styles.
 
 Returns
 -------
 
 fig            : matplotlib.Figure
                  Figure with the plots.
 axes           : matplotlib.Axes
                  Axes for the figure.
                  
 """
 
 (p,
  data,minimum_diff,maximum_diff,
  minimum_ref,maximum_ref
 ) = get_from_konrad_ref(variable,exps,exp_ref)            # Gets the data
 n = len(list(data.keys()))                                # Experiment count
 ext_exps = list(data.keys())                              # Experiment list
 
 width = 2                                                 # Width (two pairs)
 height = 5                                                # Height
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(width*2,height))
  axes = {
   "ref"  : fig.add_subplot(1,2,1),
   "diff" : fig.add_subplot(1,2,2)
  }
  
  for e in data:                                           # Plotting
   ax = axes["diff"]
   labelsd = labels
   stylesd = styles
   if e == exp_ref:
    ax = axes["ref"]
    labelsd = label_ref
    stylesd = style_ref
    if isinstance(label_ref,str):
     labelsd = { e : label_ref }
    if isinstance(style_ref,dict):
     stylesd = { e : style_ref }
   if labelsd == None:
    if stylesd == None:
     ax.plot(data[e],p)
    elif stylesd[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(data[e],p)
    elif stylesd[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(data[e],p,label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(data[e],p,label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(data[e],p,**labels_and_styles)
  
  for e in axes:
   ax=axes[e]
   
   ax.invert_yaxis()                                       # Settings for y-axes
   ax.set_yscale("log")
   if e != "ref":
    ax.tick_params(axis='y',which='both',labelleft=False)
   if e == "ref":
    formatter.set_yaxis_formatter(formatter.HectoPascalLogFormatter(),
                                  ax=ax)
    ax.tick_params(axis='y',which='both',labelsize=6)
    ax.set_ylabel(p_a,fontsize=6)
   
   if e != "ref":                                          # Settings for x-axes
    ax.set_xlim(minimum_diff - delta,maximum_diff + delta)
    ax.set_xlabel(Dvariable_label[variable],fontsize=6)
   if e == "ref":
    ax.set_xlim(minimum_ref - delta_ref,maximum_ref + delta_ref)
    ax.set_xlabel(variable_label[variable],fontsize=6)
   ax.tick_params(axis='x',which='both',labeltop=False,labelsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if e != "ref":                                          # Reference line
    ax.axvline(x=xrefline,lw=0.7,color=(0,0,0))
   if ( e == "ref" and xrefline_ref != None ):
    ax.axvline(x=xrefline_ref,lw=0.7,color=(0,0,0))
   
   if e != "ref":
    ax.legend(fontsize=5)                                  # Legend
   if ( e == "ref" and label_ref != None ):
    ax.set_title(label_ref,fontsize=5)
 
 else:
  
  fig = plt.figure(dpi=300,figsize=(width*n,height))
  axes = { ext_exps[i-1] : fig.add_subplot(1,n,i) for i in range(1,n+1) }
  
  for e in axes:                                           # Plotting
   ax = axes[e]
   labelsd = labels
   stylesd = styles
   if e == exp_ref:
    labelsd = label_ref
    stylesd = style_ref
    if isinstance(label_ref,str):
     labelsd = { e : label_ref }
    if isinstance(style_ref,dict):
     stylesd = { e : style_ref }
   if labelsd == None:
    if stylesd == None:
     ax.plot(data[e],p)
    elif stylesd[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(data[e],p)
    elif stylesd[e] == None:
     ax.plot(data[e],p)
    else:
     ax.plot(data[e],p,**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(data[e],p,label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(data[e],p,label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(data[e],p,**labels_and_styles)
     
   ax.invert_yaxis()                                       # Settings for y-axes
   ax.set_yscale("log")
   if e != exp_ref:
    ax.tick_params(axis='y',which='both',labelleft=False)
   if e == exp_ref:
    formatter.set_yaxis_formatter(formatter.HectoPascalLogFormatter(),
                                  ax=ax)
    ax.tick_params(axis='y',which='both',labelsize=6)
    ax.set_ylabel(p_a,fontsize=6)
   
   if e != exp_ref:                                        # Settings for x-axes
    ax.set_xlim(minimum_diff - delta,maximum_diff + delta)
   if e == exp_ref:
    ax.set_xlim(minimum_ref - delta_ref,maximum_ref + delta_ref)
   if e != exp_ref:
    ax.set_xlabel(Dvariable_label[variable],fontsize=6)
   if e == exp_ref:
    ax.set_xlabel(variable_label[variable],fontsize=6)
   ax.tick_params(axis='x',which='both',labeltop=False,labelsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if e != exp_ref:                                        # Reference line
    ax.axvline(x=xrefline,lw=0.7,color=(0,0,0))
   if ( e == exp_ref and xrefline_ref != None ):
    ax.axvline(x=xrefline_ref,lw=0.7,color=(0,0,0))
   
   if e != exp_ref:                                        # Set axes titles
    ax.set_title(labels[e],fontsize=5)
   if ( e == exp_ref and label_ref != None ):
    ax.set_title(label_ref,fontsize=5)
 
 if title != None:                                         # Title settings
  fig.suptitle(title,fontsize=5,y=0.95)
 
 fig.set_tight_layout("tight")                             # Other settings
 
 return (fig,axes)

def plot_from_konrad_ts1_diff(variable,exps,exp_ref,
                              same=False,
                              delta=0.5,
                              delta_ref=0.5,
                              yrefline=0,
                              yrefline_ref=None,
                              title=None,
                              labels=None,
                              styles=None,
                              label_ref=None,
                              style_ref=None
                             ):
 """Plots time series of convective top height from several `konrad`
 experiments.
 
 Parameters
 ----------
 
 variable       : str
                  Variable name in the output file.
 exps           : list
                  List of strings describing the paths to the output files.
 same           : bool, optional
                  If `True`, profiles are plotted in the same pair of axes.
                  Otherwise, there are as many pairs of axes as profiles
                  (`False` is the default).
 delta          : scalar, optional
                  Offset added to the extrema of the data.
 delta_ref      : scalar, optional
                  Offset added to the extrema of the reference data.
 yrefline       : scalar or `None`, optional
                  If `same` is `True`, plots a horizontal reference line at the
                  given scalar value. If `same` is `False`, plots vertical
                  reference lines at each axis. By default the lines are zero
                  lines.
 yrefline_ref   : scalar or `None` optional
                  Plots a horizontal reference line at the given scalar value.
                  If `None`, no reference line is plotted: this is the default.
 title          : str or `None`, optional
                  An overarching title for the figure (`None` is the default).
 labels, styles : dict or `None`, optional
                  Dictionaries containing profile labels and line styles.
                  Usually created by zipping two lists, e.g.
                  `labels = dict(zip(exps,labels))`. The `styles` dictionaries
                  are passed as keyword arguments to the `plt.plot` function.
                  If `None`, one gets default styles and/or labels.
 label_ref      : str or `None`, optional
                  String containing reference profile label.
 style_ref      : dict or `None`, optional
                  Dictionary containing reference profile line styles.
 
 Returns
 -------
 
 fig            : matplotlib.Figure
                  Figure with the plots.
 axes           : matplotlib.Axes
                  Axes for the figure.
                  
 """
 (times,t_min,t_max,
  data,minimum_diff,maximum_diff,
  minimum_ref,maximum_ref
 ) = get_from_konrad_ref(variable,exps,exp_ref)            # Gets the data
 n = len(list(data.keys()))                                # Experiment count
 ext_exps = list(data.keys())                              # Experiment list
 
 width = 5                                                 # Width
 height = 2                                                # Height (two pairs)
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(width,3*height))
  axes = {
   "ref"  : fig.add_subplot(2,1,1),
   "diff" : fig.add_subplot(2,1,2)
  }
  
  for e in data:                                           # Plotting
   ax = axes["diff"]
   labelsd = labels
   stylesd = styles
   if e == exp_ref:
    ax = axes["ref"]
    labelsd = label_ref
    stylesd = style_ref
    if isinstance(label_ref,str):
     labelsd = { e : label_ref }
    if isinstance(style_ref,dict):
     stylesd = { e : style_ref }
   if labelsd == None:
    if stylesd == None:
     ax.plot(times[e],data[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(times[e],data[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(times[e],data[e],label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e],label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(times[e],data[e],**labels_and_styles)
  
  for e in axes:
   ax=axes[e]
   
   if e == "ref":                                          # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e != "ref":
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel(t_a,fontsize=6)
   
   if e != "ref":                                          # Settings for y-axes
    ax.set_ylim(minimum_diff - delta,maximum_diff + delta)
    ax.set_ylabel(Dvariable_label[variable],fontsize=6)
   if e == "ref":
    ax.set_ylim(minimum_ref - delta_ref,maximum_ref + delta_ref)
    ax.set_ylabel(variable_label[variable],fontsize=6)
   ax.tick_params(axis='y',which='both',labelsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if e != "ref":                                          # Reference line
    ax.axhline(y=yrefline,lw=0.7,color=(0,0,0))
   if ( e == "ref" and yrefline_ref != None ):
    ax.axhline(y=yrefline_ref,lw=0.7,color=(0,0,0))
   
   if e != "ref":
    ax.legend(fontsize=5)                                  # Legend
   if ( e == "ref" and label_ref != None ):
    ax.set_title(label_ref,fontsize=5)
    
 else:
  
  fig = plt.figure(dpi=300,figsize=(width,n*height))
  axes = { ext_exps[i-1] : fig.add_subplot(n,1,i) for i in range(1,n+1) }
  
  for e in axes:                                           # Plotting
   ax = axes[e]
   labelsd = labels
   stylesd = styles
   if e == exp_ref:
    labelsd = label_ref
    stylesd = style_ref
    if isinstance(label_ref,str):
     labelsd = { e : label_ref }
    if isinstance(style_ref,dict):
     stylesd = { e : style_ref }
   if labelsd == None:
    if stylesd == None:
     ax.plot(times[e],data[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(times[e],data[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e])
    else:
     ax.plot(times[e],data[e],**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(times[e],data[e],label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(times[e],data[e],label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(times[e],data[e],**labels_and_styles)
   
   if e != exp_ref:                                        # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e == exp_ref:
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel(t_a,fontsize=6)
   
   if e != exp_ref:                                        # Settings for y-axes
    ax.set_ylim(minimum_diff - delta,maximum_diff + delta)
    ax.set_ylabel(Dvariable_label[variable],fontsize=6)
   if e == exp_ref:
    ax.set_ylim(minimum_ref - delta_ref,maximum_ref + delta_ref)
    ax.set_ylabel(variable_label[variable],fontsize=6)
   ax.tick_params(axis='y',which='both',labelsize=6)
   
   ax.minorticks_off()                                     # Unsets minor ticks
   
   if e != exp_ref:                                        # Reference line
    ax.axhline(y=yrefline,lw=0.7,color=(0,0,0))
   if ( e == exp_ref and yrefline_ref != None ):
    ax.axhline(y=yrefline_ref,lw=0.7,color=(0,0,0))
   
   if e != exp_ref:                                        # Set axes titles
    ax.set_title(labels[e],fontsize=5)
   if ( e == exp_ref and label_ref != None ):
    ax.set_title(label_ref,fontsize=5)
   
 if title != None:                                         # Title settings
  fig.suptitle(title,fontsize=5,y=0.95)
 
 fig.set_tight_layout("tight")                             # Other settings
 
 return (fig,axes)