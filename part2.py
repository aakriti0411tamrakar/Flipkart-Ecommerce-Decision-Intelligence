"""Part 2: deterministic, from-scratch churn classification and evaluation."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

def load_split():
    with (ROOT / 'data' / 'records.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    names = ['customer_tenure_months','complaint_count_90d','discount_order_share','avg_order_value_inr_hundreds','loyalty_score','cod_flag','plus_member_flag']
    X = np.array([[float(r[c]) for c in names] for r in rows], dtype=float)
    y = np.array([int(r['churned']) for r in rows], dtype=int)
    train = np.sort(np.concatenate([np.flatnonzero(y == c)[:(7 * int(np.sum(y == c))) // 10] for c in (0, 1)]))
    test = np.sort(np.concatenate([np.flatnonzero(y == c)[(7 * int(np.sum(y == c))) // 10:] for c in (0, 1)]))
    mean, sd = X[train].mean(axis=0), X[train].std(axis=0, ddof=1)
    return (X[train]-mean)/sd, y[train], (X[test]-mean)/sd, y[test]

def sigmoid(z):
    return np.exp(-np.logaddexp(0., -z))

def loss(X,y,w,b):
    z = X @ w + b
    return float(np.mean(np.logaddexp(0., z) - y*z))

def fit_logistic(X,y,eta=0.1,epochs=1000):
    w = np.zeros(X.shape[1]); b = 0.
    history = {0: loss(X,y,w,b)}
    for epoch in range(1, epochs+1):
        err = sigmoid(X@w+b)-y
        dw = X.T@err/len(y); db = float(err.mean())
        w -= eta*dw; b -= eta*db
        if epoch in (1,10,100,200,1000): history[epoch] = loss(X,y,w,b)
    return w,b,history

def confusion(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp=int(np.sum((pred==1)&(y==1))); fp=int(np.sum((pred==1)&(y==0)))
    tn=int(np.sum((pred==0)&(y==0))); fn=int(np.sum((pred==0)&(y==1)))
    return tp,fp,tn,fn

def metrics(c):
    tp,fp,tn,fn = c
    div=lambda a,b: float(a/b) if b else 0.
    p=div(tp,tp+fp); r=div(tp,tp+fn)
    return dict(accuracy=div(tp+tn,sum(c)),precision=p,recall=r,f1=div(2*p*r,p+r),specificity=div(tn,tn+fp))

def knn_probs(reference_X, reference_y, queries, K):
    # Stable sort preserves reference file order on exact distance ties.
    result = []
    for x in queries:
        distances = np.sum((reference_X-x)**2,axis=1)
        closest = np.argsort(distances,kind='stable')[:K]
        result.append(float(np.mean(reference_y[closest])))
    return np.array(result)

def write_csv(name,header,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.writer(f); w.writerow(header)
        for row in rows: w.writerow([f'{v:.4f}' if isinstance(v,(float,np.floating)) else v for v in row])

def main():
    OUT.mkdir(exist_ok=True)
    X,y,Xtest,ytest = load_split()
    assert (len(y),int(y.sum()),len(ytest),int(ytest.sum()))==(560,119,240,51)
    r={}
    r.update(train_rows=len(y),train_positives=int(y.sum()),test_rows=len(ytest),test_positives=int(ytest.sum()))
    w,b,h=fit_logistic(X,y)
    for epoch in (0,1,10,100,1000):r[f'loss_epoch_{epoch}']=h[epoch]
    for eta in (0.01,0.1,1.0):
        _,_,hist=fit_logistic(X,y,eta,200)
        r[f'loss_epoch_200_eta_{eta}']=hist[200]
    for j in range(7):
        r[f'weight_X{j+1}']=w[j]
    r['bias']=b
    r['test_bce']=loss(Xtest,ytest,w,b)
    for j in range(7):r[f'odds_ratio_X{j+1}']=np.exp(w[j])
    prob=sigmoid(Xtest@w+b)
    for tau in (0.5,0.3):
        prefix=f'tau_{tau}_'
        c=confusion(ytest,prob>=tau)
        for key,val in zip(('tp','fp','tn','fn'),c):r[prefix+key]=val
        for key,val in metrics(c).items():r[prefix+key]=val
    r['blind_accuracy']=float(np.mean(ytest==0))
    sweep=[]
    for i in range(1,10):
        tau=i/10; c=confusion(ytest,prob>=tau)
        sweep.append((tau,*c,metrics(c)['recall'],1-metrics(c)['specificity']))
    write_csv('part2_thresholds.csv',['tau','tp','fp','tn','fn','tpr','fpr'],sweep)
    cv=[]
    for K in range(1,22,2):
        fold_scores=[]
        for fold in range(5):
            validation=np.arange(len(y))%5==fold
            predictions=knn_probs(X[~validation],y[~validation],X[validation],K)>=0.5
            fold_scores.append(metrics(confusion(y[validation],predictions))['f1'])
        cv.append((K,float(np.mean(fold_scores))))
    write_csv('part2_knn_cv.csv',['K','mean_cv_f1'],cv)
    bestK=max(cv,key=lambda pair:(pair[1],-pair[0]))[0]
    r['knn_best_k']=bestK
    c=confusion(ytest,knn_probs(X,y,Xtest,bestK)>=0.5)
    for key,val in zip(('tp','fp','tn','fn'),c):r['knn_'+key]=val
    for key in ('accuracy','precision','recall','f1'):r['knn_'+key]=metrics(c)[key]
    train_prob=sigmoid(X@w+b)
    costs=[]
    for k in range(1,20):
        tau=k/20; tp,fp,tn,fn=confusion(y,train_prob>=tau)
        costs.append((50*fp+400*fn,tau))
    mincost,tau=min(costs)
    r['cost_tau']=tau; r['cost_train_at_tau']=mincost
    def cost(c):return 50*c[1]+400*c[3]
    r['cost_test_at_tau']=cost(confusion(ytest,prob>=tau))
    r['cost_test_at_0.5']=cost(confusion(ytest,prob>=0.5))
    r['cost_test_flag_nobody']=400*int(ytest.sum())
    with (OUT/'part2_results.txt').open('w') as f:
        for key,val in r.items():
            f.write(f'{key} = {int(val) if isinstance(val,(int,np.integer)) else f"{val:.4f}"}\n')
    print('Part 2 complete:',len(r),'keys; best K',bestK,'cost threshold',tau)
    return r

if __name__=='__main__':main()
