"""Part 4: hand-coded split criteria, ensemble voting, AdaBoost and parameter counts."""
import csv
from pathlib import Path
import numpy as np
from part2 import load_split, fit_logistic, sigmoid, knn_probs, confusion, metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'

def write_csv(name,header,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header)
        for row in rows:
            w.writerow([f'{float(v):.4f}' if isinstance(v,(float,np.floating)) else v for v in row])

def impurity(y,kind):
    if not len(y):return 0.
    p=float(np.mean(y))
    if kind=='entropy':return -sum(q*np.log2(q) for q in (p,1-p) if q>0)
    return 1-p*p-(1-p)**2

def split_mask(X,name):
    if name=='S1':return X[:,5]==1
    if name=='S2':return X[:,6]==1
    if name=='S3':return X[:,2]>0.5
    if name=='S4':return X[:,0]<=24
    raise ValueError(name)

def cost(c):return 50*c[1]+400*c[3]

def main():
    OUT.mkdir(exist_ok=True)
    X,y,Xt,yt=load_split()
    # Undo standardisation for original-unit split thresholds, using source records.
    with (ROOT/'data/records.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    fields=['customer_tenure_months','complaint_count_90d','discount_order_share','avg_order_value_inr_hundreds','loyalty_score','cod_flag','plus_member_flag']
    original=np.array([[float(r[c]) for c in fields] for r in rows])
    labels=np.array([int(r['churned']) for r in rows])
    train=np.sort(np.concatenate([np.flatnonzero(labels==c)[:7*int(np.sum(labels==c))//10] for c in (0,1)]))
    test=np.sort(np.concatenate([np.flatnonzero(labels==c)[7*int(np.sum(labels==c))//10:] for c in (0,1)]))
    assert np.array_equal(labels[train],y) and np.array_equal(labels[test],yt)
    r={'root_entropy':impurity(y,'entropy'),'root_gini':impurity(y,'gini')}
    split_rows=[]
    for name in ('S1','S2','S3','S4'):
        mask=split_mask(original[train],name)
        gains=[]
        for kind in ('entropy','gini'):
            child=(len(y[mask])*impurity(y[mask],kind)+len(y[~mask])*impurity(y[~mask],kind))/len(y)
            gains.append(impurity(y,kind)-child)
        split_rows.append((name,int(mask.sum()),int(y[mask].sum()),int((~mask).sum()),int(y[~mask].sum()),*gains))
    write_csv('part4_splits.csv',['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'],split_rows)
    best_info=max(split_rows,key=lambda z:z[5])[0];best_gini=max(split_rows,key=lambda z:z[6])[0]
    r['best_split_by_info_gain']=best_info;r['best_split_by_gini_gain']=best_gini
    mask=split_mask(original[train],best_info)
    yes=float(np.mean(y[mask]));no=float(np.mean(y[~mask]))
    r['stump_rate_yes']=yes;r['stump_rate_no']=no
    w,b,_=fit_logistic(X,y)
    logistic=sigmoid(Xt@w+b)
    with (OUT/'part2_results.txt').open() as f:
        part2=dict(line.strip().split(' = ',1) for line in f if ' = ' in line)
    k=int(part2['knn_best_k'])
    knn=knn_probs(X,y,Xt,k)
    stump=np.where(split_mask(original[test],best_info),yes,no)
    lp=logistic>=0.3;kp=knn>=0.5;sp=stump>=0.3
    predictions=[lp,kp,sp,(lp.astype(int)+kp.astype(int)+sp.astype(int))>=2,(logistic+knn+stump)/3>=0.3]
    vote_rows=[]
    for name,pred in zip(('logistic','knn','stump','hard_vote','soft_vote'),predictions):
        cells=confusion(yt,pred);met=metrics(cells)
        vote_rows.append((name,*cells,met['precision'],met['recall'],met['f1']))
    write_csv('part4_voting.csv',['model','tp','fp','tn','fn','precision','recall','f1'],vote_rows)
    truth=np.array([1,1,1,-1,-1,-1,1,-1])
    h1=np.array([1,1,-1,-1,-1,-1,-1,-1]);h2=np.array([-1,1,1,-1,1,-1,1,-1])
    weights=np.ones(8)/8
    boost_rows=[];alphas=[]
    for t,h in enumerate((h1,h2),1):
        err=float(np.sum(weights[h!=truth]));alpha=0.5*np.log((1-err)/err)
        unnorm=weights*np.exp(-alpha*truth*h);after=unnorm/unnorm.sum()
        r[f'round{t}_error']=err;r[f'round{t}_alpha']=float(alpha)
        boost_rows.extend((t,i+1,float(weights[i]),float(unnorm[i]),float(after[i])) for i in range(8))
        weights=after;alphas.append(alpha)
    write_csv('part4_adaboost.csv',['round','point','weight_before','weight_unnormalised','weight_after'],boost_rows)
    score=alphas[0]*h1+alphas[1]*h2
    r['ensemble_accuracy']=float(np.mean(np.sign(score)==truth))
    r['oob_fraction']=(1-1/560)**560
    # convolution: 28x28x1 -> 28x28x6 -> 14x14x6 -> 10x10x16 -> 5x5x16 -> 400 -> 32 -> 10
    arch=[('mlp_hidden1',(784+1)*16),('mlp_hidden2',(16+1)*16),('mlp_output',(16+1)*10),
          ('cnn_conv1',(5*5*1+1)*6),('cnn_conv2',(5*5*6+1)*16),('cnn_dense1',(5*5*16+1)*32),('cnn_dense2',(32+1)*10)]
    write_csv('part4_architecture.csv',['layer','parameters'],arch)
    r['mlp_total']=sum(v for _,v in arch[:3]);r['cnn_total']=sum(v for _,v in arch[3:])
    r['fc_layer_params']=(128*128*3+1)*1024
    r['conv_layer_params']=(3*3*3+1)*64
    r['rnn_total']=128*64+128*128+128+16*128+16
    with (OUT/'part4_results.txt').open('w') as f:
        for key,value in r.items():
            if isinstance(value,str):v=value
            elif isinstance(value,(int,np.integer)):v=str(value)
            else:v=f'{value:.4f}'
            f.write(f'{key} = {v}\n')
    print('Part 4:',r)
    print('Split rows:',split_rows)
    print('Voting rows:',vote_rows)
    print('Boost rows:',boost_rows)
    return r,split_rows,vote_rows,boost_rows,arch

if __name__=='__main__':main()
