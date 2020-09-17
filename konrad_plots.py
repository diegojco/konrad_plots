"""A module that helps to plot the results of `konrad`"""

import numpy as np
from netCDF4 import Dataset as ncload
import matplotlib.pyplot as plt
from typhon.plots import formatter

# Constants

## String constants

### Units

p_u       = r"\mathrm{hPa}"
T_u       = r"\mathrm{K}"
dTdz_u    = r"\mathrm{K}\,\mathrm{km}^{-1}"
I_u       = r"\mathrm{W}\,\mathrm{m}^{-2}"
sigma_u   = r"1\times 10^{-6}"
ct_u      = r"\mathrm{km}"

### Quantity symbols

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

### Difference symbols

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

### Symbols with units for axes labelling

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
ct_a      = r"$"+ct_s+r"/"+ct_u+r"$"

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
Dct_a     = r"$"+Dct_s+r"/"+ct_u+r"$"

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
 "O3"    : lambda datae: identity(datae,factor=10**6),
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
}

# Extraction functions

def extract(group,variable,file,
            idx=-1
           ):
 """Extracts a variable from a group in the `konrad` output files, by default it
 takes the last time available.
 
 Parameters
 ----------
 
 group    : str
            Group name in the output file.
 variable : str
            Variable name in the output file.
 file     : str
            Path to the output file.
 idx      : int, optional
            The required index in the time dimension (the last is the default).
 
 Returns
 -------
 
 numpy.ndarray
            The profile at index time `idx` of the requested variable.
            
 """
 with ncload(file,mode="r") as f:
  temp = f.groups[group][variable][idx,:]
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

def extract_ct(file,
               idxi=0,
               idxf=-1
              ):
 """Extracts a time series of the convective top from `konrad` output.
 
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
 
 ct   : numpy.ndarray
        Time series of the convective top.
            
 """
 with ncload(file,mode="r") as f:
  temp = f.groups["convection"]["convective_top_height"][idxi:idxf]
  temp = np.array(temp,dtype="float64")
  return temp

def get_from_konrad(variable,exps,
                    idx=-1
                   ):
 """Extracts a variable from the output files of several `konrad` experiments.
 
 Parameters
 ----------
 
 variable : str
            Variable name in the output file.
 exps     : list
            List of strings describing the paths to the output files.
 idx      : int, optional
            The required index in the time dimension (the last is the default).
 
 Returns
 -------
 
 p        : numpy.ndarray
            Pressure levels corresponding to the variable.
 data     : dict
            The requested profiles of the variable at index time `idx` for all
            the experiments in `exps`.
 minimum  : scalar
            The minimum of all profiles.
 maximum  : scalar
            The maximum of all profiles.
            
 """
 
 if variable in ["temp","lapse","O3"]:               # Selecting pressure levels
  p = extract_plev(exps[0])
 else:
  p = extract_plev(exps[0],plevtype="phlev")
 
 data = [ [ extract(rule[0],rule[1],e,idx=idx)
            for rule in ext_def[variable] ]
          for e in exps ]
 data = [ clc_def[variable](e) for e in data ]
 data = dict(zip(exps,data))
 
 minimum = np.array([ data[e].min() for e in data ]).min()
 maximum = np.array([ data[e].max() for e in data ]).max()
 return (p,data,minimum,maximum)

def get_from_konrad_ct(exps,
                       idxi=0,
                       idxf=-1
                      ):
 """Extracts time series of the convective top height in kilometers.
 
 Parameters
 ----------
 
 exps : list
        List of strings describing the paths to the output files.
 idxi : int, optional
        The initial index in the time dimension (the first is the default).
 idxf : int, optional
        The final index in the time dimension (the last is the default).
 
 Returns
 -------
 
 ct   : numpy.ndarray
        Time series of the convective top.
            
 """
 
 ct = [ extract_ct(e,idxi=idxi,idxf=idxf) / 1000
        for e in exps ]
 ct = dict(zip(exps,ct))
 
 minimum = np.array([ ct[e].min() for e in ct ]).min()
 maximum = np.array([ ct[e].max() for e in ct ]).max()
 return (ct,minimum,maximum)

def get_from_konrad_ref(variable,exps,exp_ref,
                        idx=-1
                       ):
 """Extracts a variable from the output files of several `konrad` experiments
 and calculates their differences from a reference experiment.
 
 Parameters
 ----------
 
 variable     : str
                Variable name in the output file.
 exps         : list
                List of strings describing the paths to the output files.
 exp_ref      : str
                Path to the output file of the reference experiment. 
 idx          : int, optional
                The required index in the time dimension (the last is the
                default).
 
 Returns
 -------
 
 p            : numpy.ndarray
                Pressure levels corresponding to the variable.
 data_diff    : dict
                The requested profiles of the variable at index time `idx` for
                all the experiments in `exps`. They are difference profiles.
 minimum_diff : scalar
                The minimum of all profiles.
 maximum_diff : scalar
                The maximum of all profiles.
 data_ref     : numpy.ndarray
                The requested profile of the variable at index time `idx` for
                `exp_ref`.
 minimum_ref  : scalar
                The reference minimum.
 maximum_ref  : scalar
                The reference maximum.
                
 """
 
 if variable in ["temp","lapse","O3"]:               # Selecting pressure levels
  p = extract_plev(exps[0])
 else:
  p = extract_plev(exps[0],plevtype="phlev")
 
 data_ref = [ extract(rule[0],rule[1],exp_ref,idx=idx)
              for rule in ext_def[variable] ]
 data_ref = clc_def[variable](data_ref)
 
 data = [ [ extract(rule[0],rule[1],e,idx=idx)
            for rule in ext_def[variable] ]
          for e in exps ]
 data = [ clc_def[variable](e) for e in data ]
 data = [ e - data_ref for e in data ]
 data = dict(zip(exps,data))
 
 minimum_diff = np.array([ data[e].min() for e in data ]).min()
 maximum_diff = np.array([ data[e].max() for e in data ]).max()
 minimum_ref = data_ref.min()
 maximum_ref = data_ref.max()
 
 data[exp_ref] = data_ref
 
 return (p,
         data,minimum_diff,maximum_diff,
         minimum_ref,maximum_ref
        )

def get_from_konrad_ct_ref(exps,exp_ref,
                           idxi=0,
                           idxf=-1
                          ):
 """Extracts time series of the convective top height in kilometers and compares
 with a reference experiment.
 
 Parameters
 ----------
 
 exps    : list
           List of strings describing the paths to the output files.
 exp_ref : list
           Path to the output file of the reference experiment.
 idxi    : int, optional
           The initial index in the time dimension (the first is the default).
 idxf    : int, optional
           The final index in the time dimension (the last is the default).
 
 Returns
 -------
 
 ct      : numpy.ndarray
           Time series of the convective top.
            
 """
 
 ct_ref = extract_ct(exp_ref,idxi=idxi,idxf=idxf) / 1000
 
 ct = [ extract_ct(e,idxi=idxi,idxf=idxf) / 1000
        for e in exps ]
 ct = [ e - ct_ref[-1] for e in ct ]
 ct = dict(zip(exps,ct))
 
 minimum_diff = np.array([ ct[e].min() for e in ct ]).min()
 maximum_diff = np.array([ ct[e].max() for e in ct ]).max()
 minimum_ref = ct_ref.min()
 maximum_ref = ct_ref.max()
 
 ct[exp_ref]=ct_ref
 return (ct,minimum_diff,maximum_diff,
         minimum_ref,maximum_ref)

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

def plot_from_konrad_ct(exps,
                        idxi=0,
                        idxf=-1,
                        same=False,
                        delta=0.5,
                        yrefline=None,
                        title=None,
                        labels=None,
                        styles=None
                       ):
 """Plots time series of convective top height from several `konrad`
 experiments.
 
 Parameters
 ----------
 
 exps           : list
                  List of strings describing the paths to the output files.
 idxi           : int, optional
                  The initial index in the time dimension (the first is the
                  default).
 idxf           : int, optional
                  The final index in the time dimension (the last is the
                  default).
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
 
 (ct,
  minimum,
  maximum
 ) = get_from_konrad_ct(exps,idxi=idxi,idxf=idxf)          # Gets the data
 n = len(exps)                                             # Experiment count
 
 times = { e : list(range(len(ct[e]))) for e in ct }       # Get times
 
 width = 5                                                 # Width
 height = 1                                                # Height (one pair)
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(width,3*height))
  axes = fig.add_subplot(1,1,1)
  
  for e in ct:                                             # Plotting
   ax = axes
   if labels == None:
    if styles == None:
     ax.plot(times[e],ct[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(times[e],ct[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**styles[e])
   else:
    if styles == None:
     ax.plot(times[e],ct[e],label=labels[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e],label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(times[e],ct[e],**labels_and_styles)
  
  ax = axes
  ax.tick_params(axis='x',which='both',
                 labeltop=False,labelsize=6)               # Settings for x-axis
  ax.set_xlabel("t",fontsize=6)
  
  ax.set_ylim(minimum - delta,maximum + delta)             # Settings for y-axis
  ax.tick_params(axis='y',which='both',labelright=False,labelsize=6)
  ax.set_ylabel(variable_label["ct"],fontsize=6)
  
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
     ax.plot(times[e],ct[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**styles[e])
   elif labels[e] == None:
    if styles == None:
     ax.plot(times[e],ct[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**styles[e])
   else:
    if styles == None:
     ax.plot(times[e],ct[e],label=labels[e])
    elif styles[e] == None:
     ax.plot(times[e],ct[e],label=labels[e])
    else:
     labels_and_styles = styles[e]
     labels_and_styles["label"] = labels[e]
     ax.plot(times[e],ct[e],**labels_and_styles)
   
   if e != exps[-1]:                                       # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e == exps[-1]:
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel("t",fontsize=6)
   
   ax.set_ylim(minimum - delta,maximum + delta)            # Settings for y-axes
   ax.tick_params(axis='y',which='both',labelsize=6)
   ax.set_ylabel(variable_label["ct"],fontsize=6)
   
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

def plot_from_konrad_ct_diff(exps,exp_ref,
                             idxi=0,
                             idxf=-1,
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
 
 exps           : list
                  List of strings describing the paths to the output files.
 idxi           : int, optional
                  The initial index in the time dimension (the first is the
                  default).
 idxf           : int, optional
                  The final index in the time dimension (the last is the
                  default).
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
 
 (ct,
  minimum_diff,maximum_diff,
  minimum_ref,maximum_ref
 ) = get_from_konrad_ct_ref(exps,exp_ref,
                            idxi=idxi,idxf=idxf)           # Gets the data
 n = len(list(ct.keys()))                                  # Experiment count
 ext_exps = list(ct.keys())                                # Experiment list
 
 times = { e : list(range(len(ct[e]))) for e in ct }       # Get times
 
 width = 5                                                 # Width
 height = 2                                                # Height (two pairs)
 
 if same:
  
  fig = plt.figure(dpi=300,figsize=(width,3*height))
  axes = {
   "ref"  : fig.add_subplot(2,1,1),
   "diff" : fig.add_subplot(2,1,2)
  }
  
  for e in ct:                                             # Plotting
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
     ax.plot(times[e],ct[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(times[e],ct[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(times[e],ct[e],label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e],label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(times[e],ct[e],**labels_and_styles)
  
  for e in axes:
   ax=axes[e]
   
   if e == "ref":                                          # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e != "ref":
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel("t",fontsize=6)
   
   if e != "ref":                                          # Settings for y-axes
    ax.set_ylim(minimum_diff - delta,maximum_diff + delta)
    ax.set_ylabel(Dvariable_label["ct"],fontsize=6)
   if e == "ref":
    ax.set_ylim(minimum_ref - delta_ref,maximum_ref + delta_ref)
    ax.set_ylabel(variable_label["ct"],fontsize=6)
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
     ax.plot(times[e],ct[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**stylesd[e])
   elif labelsd[e] == None:
    if stylesd == None:
     ax.plot(times[e],ct[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e])
    else:
     ax.plot(times[e],ct[e],**stylesd[e])
   else:
    if stylesd == None:
     ax.plot(times[e],ct[e],label=labelsd[e])
    elif stylesd[e] == None:
     ax.plot(times[e],ct[e],label=labelsd[e])
    else:
     labels_and_styles=stylesd[e]
     labels_and_styles["label"]=labelsd[e]
     ax.plot(times[e],ct[e],**labels_and_styles)
   
   if e != exp_ref:                                        # Settings for x-axes
    ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
   if e == exp_ref:
    ax.tick_params(axis='x',which='both',labelsize=6)
    ax.set_xlabel("t",fontsize=6)
   
   if e != exp_ref:                                        # Settings for y-axes
    ax.set_ylim(minimum_diff - delta,maximum_diff + delta)
    ax.set_ylabel(Dvariable_label["ct"],fontsize=6)
   if e == exp_ref:
    ax.set_ylim(minimum_ref - delta_ref,maximum_ref + delta_ref)
    ax.set_ylabel(variable_label["ct"],fontsize=6)
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