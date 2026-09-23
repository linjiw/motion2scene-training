import numpy as np
from sim_indep import run
kw=dict(sig_task=1.5, sig_int=0.5, sig_inst=1.0, n_long_zero=6)
print("type-I (pA=pB=5.5/19):")
for S in [3,5,8]:
    r=run(S,pA=5.5/19,pB=5.5/19,reps=1000,**kw); print(S, {k:round(v,3) for k,v in r.items() if k!='gate_pass'})
print("gate 'mean>=5/19 over S seeds' pass prob for arm A true rate:")
for pA in [3/19,4/19,5/19,6/19,7/19]:
    out=[]
    for S in [3,5,8]:
        r=run(S,pA=pA,pB=pA,reps=600,**kw); out.append((S,round(r['gate_pass'],2)))
    print(round(pA*19,1), out)
print("power detect 7 vs 2 (approach c2 vs uniform c2 as observed):")
for S in [1,3,5]:
    r=run(S,pA=7/19,pB=2/19,reps=1000,**kw); print(S, {k:round(v,2) for k,v in r.items() if k!='gate_pass'})
print("power detect 5 vs 3:")
for S in [5,10,15,20]:
    r=run(S,pA=5/19,pB=3/19,reps=600,**kw); print(S, {k:round(v,2) for k,v in r.items() if k!='gate_pass'})
