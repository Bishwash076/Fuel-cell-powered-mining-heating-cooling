"""Advanced 2-D PEMFC sensitivity study; NumPy + Matplotlib only.
Electrochemical expressions retained from user-supplied model. H2/O2/water
balance from Faraday and reaction stoichiometry. Original empirical P_O2*
NOT coupled to oxygen feed: lambda affects air-flow only, NOT voltage.
All plots are deterministic model sensitivities, NOT empirical distributions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colors

F=96485.; MH2=.002016; MO2=.031998; MH2O=.018015; MN2=.028014
LHV=120.1e6; NC=500; NS=8; AREA=450.; THICK=.0125
H2FEED=30.; UTIL=.90; LAMBDA=2.; XAIR=.21
T0=353.15; P0=3.; XWAN=.10; XWCAT=.20
XI1=-.9514; XI2=.00312; XI3=7.4e-5; XI4=-1.87e-4; ND=-.23

def psat(T):
    tc=T-273.15
    return 10**(-2.1794+.02953*tc-9.1837e-5*tc**2+1.4454e-7*tc**3)

def partials(T,j):
    h2=.5*psat(T)*(1/(np.exp(1.653*j/T**1.334)*XWAN)-1)
    o2=psat(T)*(1/(np.exp(4.192*j/T**1.334)*XWCAT)-1)
    return np.maximum(h2,.01),np.maximum(o2,.01)

def components(T,j,area=AREA,thick=THICK):
    ph,po=partials(T,j)
    co=(po*101325)/(5.08e6*np.exp(-498/T))*1e-6 # supplied equation, unchanged
    E=1.229-.85e-3*(T-298.15)+4.3085e-5*T*(np.log(ph)+.5*np.log(po))
    act=XI1+XI2*T+XI3*T*np.log(co)+XI4*T*np.log(j*area)
    rho=181.6*(1+.03*j+.062*(T/303)**2*j**2.5)/((14-.634-3*j)*np.exp(4.18*(T-303)/T))
    ohm=-j*rho*thick
    m=np.where(np.asarray(T)>=312.15,1.1e-4-1.2e-6*(T-273.15),3.3e-3-8.2e-5*(T-273.15))
    diff=m*np.exp(ND*j)
    return E,act,ohm,diff

def voltage(T,j,area=AREA,thick=THICK):
    return np.maximum(sum(components(T,j,area,thick)),0)

def h2used(j,stacks=NS,area=AREA):
    return j*area*NC*stacks/(2*F)*MH2*3600

def jlimit(feed=H2FEED,util=UTIL,stacks=NS,area=AREA):
    return feed*util/(MH2*3600)*(2*F)/(NC*stacks*area)

def power(T,j,stacks=NS,area=AREA,thick=THICK):
    return NC*stacks*j*area*voltage(T,j,area,thick)/1000

def efficiency(T,j,area=AREA,thick=THICK):
    return 100*voltage(T,j,area,thick)/(LHV*MH2/(2*F))

def balance(j,feed=H2FEED,stacks=NS,area=AREA,lam=LAMBDA):
    used=h2used(j,stacks,area)
    if np.any(np.asarray(used)>feed+1e-8):raise ValueError('H2 use exceeds feed')
    o2used=used*MO2/(2*MH2); o2in=lam*o2used
    n2in=o2in/MO2*(1-XAIR)/XAIR*MN2
    water=used*MH2O/MH2
    return dict(h2_in=feed,h2_used=used,h2_out=feed-used,o2_used=o2used,
                o2_in=o2in,o2_out=o2in-o2used,n2_in=n2in,water=water,air=o2in+n2in)

def main():
    lim=jlimit(); jb=.85*lim
    b=balance(lim)
    inlet=b['h2_in']+b['air']; outlet=b['h2_out']+b['o2_out']+b['n2_in']+b['water']
    print('=== 30 kg/h H2 | 90% utilization | lambda_O2=2 | original empirical model ===')
    for key,value in b.items(): print(f'{key:12s} {value:12.5f} kg/h')
    print(f'Mass balance: inlet={inlet:.6f}, outlet={outlet:.6f}, error={inlet-outlet:.3e} kg/h')
    print(f'Feed-limited j={lim:.5f} A/cm2; power={power(T0,lim):.3f} kW; efficiency={efficiency(T0,lim):.3f}%')
    print('WARNING: lambda and air feed do not influence voltage in the supplied P_O2* correlation.')
    print('WARNING: nominal pressure P0 is metadata, not an independent input in the supplied voltage equations.')
    print('WARNING: original fitted coefficients, empirical concentration units and loss signs require validation.')
    print('WARNING: gross DC power; auxiliaries, water phase separation and humidification excluded.')
    j=np.linspace(.025,lim,110); ts=np.linspace(323.15,373.15,95)
    J,T=np.meshgrid(j,ts)
    V=voltage(T,J); P=power(T,J); ETA=efficiency(T,J)
    fig,axs=plt.subplots(5,2,figsize=(19,25),layout='constrained')
    a,bx,c,e,f,g,h,k,m,n=axs.flat
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10})
    # A: filled power contours with labelled isolines
    cf=a.contourf(J,T-273.15,P,45,cmap='turbo'); lines=a.contour(J,T-273.15,P,7,colors='white',linewidths=.6)
    a.clabel(lines,fontsize=7,fmt='%.0f');fig.colorbar(cf,ax=a,label='Gross power (kW)')
    a.set(title='(a) Response surface: gross power',xlabel='Current density (A/cm²)',ylabel='Temperature (°C)')
    # B: diverging percent power change vs 80 C at same current
    pref=power(T0,J); pct=100*(P-pref)/np.maximum(pref,1e-9)
    bound=max(abs(pct.min()),abs(pct.max()),.01)
    im=bx.pcolormesh(J,T-273.15,pct,cmap='coolwarm',norm=TwoSlopeNorm(vcenter=0,vmin=-bound,vmax=bound),shading='auto')
    cs=bx.contour(J,T-273.15,pct,levels=9,colors='black',linewidths=.45,alpha=.65)
    bx.clabel(cs,fontsize=7,fmt='%.1f')
    fig.colorbar(im,ax=bx,label='Change relative to 80°C (%)')
    bx.axhline(80,color='black',ls='--',lw=1)
    bx.set(title='(b) Temperature-induced power change',xlabel='Current density (A/cm²)',ylabel='Temperature (°C)')
    # C: multi-temperature voltage and power, dual axis
    for temp in [50,60,70,80,90,100]:
        c.plot(j,power(temp+273.15,j),lw=1.9,label=f'{temp}°C')
    c.axvline(lim,color='black',ls=':',label='Feed limit')
    c.set(title='(c) Polarization-to-power sensitivity',xlabel='Current density (A/cm²)',ylabel='Gross power (kW)');c.legend(ncol=2,fontsize=8)
    # E: tornado at fixed fraction of feed limit; physical feasible for every scenario
    base=float(power(T0,jb)); cases=[]
    for name,low,high in [('Temperature (°C)',323.15,373.15),('Membrane thickness (cm)',.8*THICK,1.2*THICK),('Active area (cm²)',.85*AREA,1.15*AREA),('Stacks',6,10),('H₂ utilization',.75,.95),('H₂ feed (kg/h)',25,35)]:
        def scenario(v):
            T,thick,area,stacks,util,feed=T0,THICK,AREA,NS,UTIL,H2FEED
            if name.startswith('Temperature'):T=v
            elif name.startswith('Membrane'):thick=v
            elif name.startswith('Active'):area=v
            elif name=='Stacks':stacks=int(v)
            elif name.startswith('H₂ utilization'):util=v
            else:feed=v
            jj=.85*jlimit(feed,util,stacks,area)
            return float(power(T,jj,stacks,area,thick))
        cases.append((name,scenario(low),scenario(high)))
    cases.sort(key=lambda x:abs(x[2]-x[1]),reverse=True)
    for i,(name,lo,hi) in enumerate(cases):
        e.plot([lo,hi],[i,i],lw=7,color='#8aadc5',solid_capstyle='round')
        e.scatter([lo,hi],[i,i],c=['#256b9e','#dc743d'],s=55,zorder=3)
    e.axvline(base,color='black',ls='--',label=f'Baseline {base:.1f} kW')
    e.set_yticks(range(len(cases)),[x[0] for x in cases]);e.invert_yaxis()
    e.set(title='(d) Tornado: feasible one-factor-at-a-time power',xlabel='Gross power (kW)');e.legend(fontsize=8)
    # F: heatmap: temperature vs utilization, each point at jlimit
    us=np.linspace(.55,.98,42); t2=np.linspace(50,100,40)
    UH,TH=np.meshgrid(us,t2); JH=jlimit(util=UH)
    PM=power(TH+273.15,JH)
    im=f.pcolormesh(UH,TH,PM,cmap='turbo',shading='auto')
    cs=f.contour(UH,TH,PM,levels=9,colors='black',linewidths=.55,alpha=.7)
    f.clabel(cs,fontsize=7,fmt='%.0f')
    fig.colorbar(im,ax=f,label='Gross power (kW)')
    f.scatter([UTIL],[80],marker='x',c='white',s=90,lw=2)
    f.set(title='(e) Coupled temperature–utilization sensitivity',xlabel='H₂ utilization (fraction)',ylabel='Temperature (°C)')
    # G: bubble plot: at varying utilization, bubble proportional water; color air feed
    uu=np.linspace(.35,.98,75); jj=jlimit(util=uu)
    pw=power(T0,jj); balances=[balance(float(q)) for q in jj]
    water=np.array([x['water'] for x in balances]); air=np.array([x['air'] for x in balances])
    sc=g.scatter(uu,pw,s=25+220*water/water.max(),c=air,cmap='turbo',alpha=.85,edgecolor='black',linewidth=.25)
    fig.colorbar(sc,ax=g,label='Dry air feed (kg/h)');g.axvline(UTIL,color='black',ls=':')
    g.set(title='(f) Bubble: water, air and power',xlabel='H₂ utilization (fraction)',ylabel='Gross power (kW)')
    # H: violin: model voltage over uniformly sampled feasible j, not empirical spread
    temp_groups=[50,60,70,80,90,100]
    dat=[voltage(t+273.15,j) for t in temp_groups]
    violin=h.violinplot(dat,positions=range(len(dat)),showmedians=True,showextrema=True)
    for pc in violin['bodies']:pc.set_facecolor('#6d96b5');pc.set_alpha(.7)
    h.set_xticks(range(len(dat)),[str(x) for x in temp_groups]);h.set(title='(g) Modelled voltage range across current sweep',xlabel='Temperature (°C)',ylabel='Cell voltage (V)')
    # I: feed-utilization and stack count sensitivity: fixed total feed, not stack feed
    feeds=np.linspace(15,45,55)
    for stacks in [4,6,8,10,12,16]:
        jj=jlimit(feed=feeds,stacks=stacks)
        k.plot(feeds,power(T0,jj,stacks=stacks),label=f'{stacks} stacks',lw=1.7)
    k.axvline(H2FEED,ls='--',color='black',lw=1)
    k.set(title='(h) Feed-limited stack scaling',xlabel='Total H₂ feed (kg/h)',ylabel='Gross power (kW)');k.legend(ncol=2,fontsize=8)
    # K: paired heatmap of power across stack and feed design space
    ns=np.arange(4,17,2); hf=np.linspace(15,45,45)
    design=np.array([[power(T0,jlimit(feed=float(feed),stacks=int(st)),stacks=int(st)) for feed in hf] for st in ns])
    im=m.imshow(design,origin='lower',aspect='auto',extent=[hf[0],hf[-1],ns[0]-1,ns[-1]+1],cmap='jet')
    fig.colorbar(im,ax=m,label='Gross power (kW)')
    m.contour(hf,ns,design,levels=8,colors='white',linewidths=.65,alpha=.85)
    m.set_yticks(ns)
    m.set(title='(i) Stack–feed design-space heatmap',xlabel='Total H₂ feed (kg/h)',ylabel='Number of stacks')
    # L: stacked product and exhaust flows, baseline j sweep
    jf=np.linspace(.01,lim,65)
    flows=[balance(float(q)) for q in jf]
    n.stackplot(jf,[[z['water'] for z in flows],[z['o2_out'] for z in flows],[z['n2_in'] for z in flows],[z['h2_out'] for z in flows]],labels=['Water produced','O₂ exhaust','N₂ exhaust','H₂ exhaust'],alpha=.88)
    n.set(title='(j) Exhaust/product mass-flow composition',xlabel='Current density (A/cm²)',ylabel='Outlet mass flow (kg/h)');n.legend(loc='upper left',fontsize=8)
    for ax in axs.flat:
        ax.tick_params(labelsize=9)
        if ax not in [a,bx,f,m]:ax.grid(alpha=.15)
    fig.suptitle('PEMFC Advanced 2D CFD-style Sensitivity | 30 kg/h H₂ feed | 8 stacks | Original supplied electrochemical relations',fontsize=15,fontweight='bold')
    target=Path.cwd()/'PEMFC_30kgph_Advanced_2D_Sensitivity_10Panels_CFD.png'
    fig.savefig(target,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)
    print('Figure saved:',target)

if __name__=='__main__':main()
