"""Opt-in process-local correction for the shifted senior SOP F8 only.

Do not alter benchmark objectives or old experiment files. Install before
importing a training/evaluation module so all gap consumers share the fix.
"""
from functools import lru_cache
import runpy
import sys
import numpy as np
from scipy.optimize import brentq
from . import core

@lru_cache(None)
def shifted_f8_minimum(dimension):
    # h(z)=-z*sin(sqrt(abs(z))); stationary points satisfy
    # 2*sin(t)+t*cos(t)=0 with t=sqrt(abs(z)). All positive roots lie
    # in ((n+1/2)*pi, (n+1)*pi); include both signs and every endpoint.
    offset=np.linspace(0.,500.,dimension)
    tmax=np.sqrt(1000.)
    roots=[]
    for n in range(int(tmax/np.pi)+1):
        lo=(n+.5)*np.pi;hi=(n+1)*np.pi
        if lo>tmax:break
        roots.append(brentq(lambda t:2*np.sin(t)+t*np.cos(t),lo,hi,xtol=1e-13))
    points=np.array([0., *[s*t*t for t in roots for s in (-1,1)]])
    minima=[];locations=[]
    for off in offset:
        lo,hi=-500.-off,500.-off
        z=np.r_[lo,hi,points[(points>=lo)&(points<=hi)]]
        f=-z*np.sin(np.sqrt(np.abs(z)))
        j=np.argmin(f);minima.append(f[j]);locations.append(z[j]+off)
    return float(sum(minima)),tuple(locations)

def install():
    original=core.objective_gap
    def corrected(problem_name, values, dimension):
        if problem_name.upper()!='SOP_F8':return original(problem_name,values,dimension)
        optimum,_=shifted_f8_minimum(int(dimension))
        # Small explicit numerical tolerance prevents roundoff at the minimizer.
        return np.maximum(np.asarray(values,dtype=np.float64)-optimum,1e-10)
    core.objective_gap=corrected

def main():
    target=sys.argv.pop(1)
    if '--protocol' not in sys.argv or sys.argv[sys.argv.index('--protocol')+1]!='senior':
        raise ValueError('The corrected entry point requires --protocol senior')
    install()
    runpy.run_module(target,run_name='__main__')

if __name__=='__main__':main()
