"""Part 1: descriptive inference and regression, implemented with NumPy arithmetic."""
import csv
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / 'data/records.csv').open(newline='') as file:
    rows = list(csv.DictReader(file))

def column(name):
    return np.array([float(row[name]) for row in rows], dtype=float)

x = column('avg_order_value_inr_hundreds')
y = column('order_value_inr_hundreds')
n = len(y)
result = {}

def put(key, value):
    result[key] = value

def regression(a, b):
    a_bar, b_bar = np.mean(a), np.mean(b)
    slope = np.sum((a-a_bar)*(b-b_bar))/np.sum((a-a_bar)**2)
    intercept = b_bar - slope*a_bar
    fitted = slope*a + intercept
    tss = np.sum((b-b_bar)**2)
    ess = np.sum((fitted-b_bar)**2)
    rss = np.sum((b-fitted)**2)
    return dict(slope=slope, intercept=intercept, fitted=fitted, tss=tss,
                ess=ess, rss=rss, r2=1-rss/tss, rmse=np.sqrt(rss/len(b)),
                mae=np.mean(np.abs(b-fitted)))

put('n', n)
mean, sd = np.mean(y), np.std(y, ddof=1)
se = sd/math.sqrt(n)
for key, val in [('mean_value',mean),('sd_value',sd),('se_value',se),
                 ('ci95_low',mean-1.96*se),('ci95_high',mean+1.96*se)]:
    put(key,val)
z_stat = (mean-8)/se
put('z_stat',z_stat)
put('reject_h0', 'yes' if abs(z_stat)>1.96 else 'no')
put('n_required',math.ceil((1.96*sd/0.5)**2))
raw = regression(x,y)
for key in ['slope','intercept','tss','ess','rss','r2','rmse','mae']:
    put(('reg_' if key in ['slope','intercept'] else '')+key,raw[key])
put('tss_equals_ess_plus_rss','yes' if abs(raw['tss']-raw['ess']-raw['rss'])<1e-6*raw['tss'] else 'no')
x_sd = np.std(x,ddof=1)
x_z = (x-np.mean(x))/x_sd
m = c = 0.0
put('gd_J0',np.mean(y**2))
for step in range(1,501):
    error = y-(m*x_z+c)
    m += 0.1*2*np.mean(x_z*error)
    c += 0.1*2*np.mean(error)
    if step in (1,10,100,500):
        put(f'gd_J{step}',np.mean((y-(m*x_z+c))**2))
put('gd_slope',m/x_sd)
put('gd_intercept',c-m*np.mean(x)/x_sd)
put('gd_slope_gap',abs(result['gd_slope']-raw['slope']))

def residual_halves(model, target):
    order = np.argsort(model['fitted'],kind='stable')
    residual = target-model['fitted']
    lower = np.std(residual[order[:n//2]],ddof=1)
    upper = np.std(residual[order[n//2:]],ddof=1)
    return lower,upper,upper/lower

lo,hi,ratio = residual_halves(raw,y)
for key,val in [('resid_sd_lower_raw',lo),('resid_sd_upper_raw',hi),('resid_sd_ratio_raw',ratio)]: put(key,val)
ly = np.log(y)
loglog = regression(np.log(x),ly)
for key in ('intercept','slope','r2','rmse'):
    put('loglog_'+key,loglog[key])
lo,hi,ratio = residual_halves(loglog,ly)
for key,val in [('resid_sd_lower_log',lo),('resid_sd_upper_log',hi),('resid_sd_ratio_log',ratio)]: put(key,val)
features = [np.ones(n),np.log(x)] + [column(name) for name in (
    'customer_tenure_months','complaint_count_90d','discount_order_share',
    'loyalty_score','cod_flag','plus_member_flag')]
X = np.column_stack(features)
beta = np.linalg.solve(X.T@X,X.T@ly)
for i,value in enumerate(beta): put(f'mr_beta_{i}',value)
mr_r2 = 1-np.sum((ly-X@beta)**2)/np.sum((ly-np.mean(ly))**2)
put('mr_r2',mr_r2)
put('mr_adj_r2',1-(1-mr_r2)*(n-1)/(n-8))
put('loglog_adj_r2',1-(1-loglog['r2'])*(n-1)/(n-2))

out = ROOT/'outputs'
out.mkdir(exist_ok=True)
with (out/'part1_results.txt').open('w') as file:
    for key,value in result.items():
        formatted = value if isinstance(value,str) else str(int(value)) if isinstance(value,(int,np.integer)) else f'{value:.4f}'
        file.write(f'{key} = {formatted}\n')
print('Part 1 completed:',len(result),'results')
print('mean:',f'{mean:.4f}','z:',f'{z_stat:.4f}','gradient gap:',f'{result["gd_slope_gap"]:.8f}')
