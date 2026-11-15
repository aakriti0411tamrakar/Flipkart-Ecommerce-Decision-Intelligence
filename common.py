"""Shared helpers (importing this file runs nothing)."""
import csv
import math
import os

import numpy as np


def load():
    with open(os.path.join("data", "records.csv")) as f:
        rd = csv.reader(f)
        next(rd)
        rows = list(rd)
    return np.array([[float(v) for v in r[1:]] for r in rows])


def split_and_standardise(A):
    """Stratified 70/30 split (Task 2.1). Returns raw/standardised X, y for train and test."""
    X, y = A[:, :7], A[:, 7]
    tr, te = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        k = (7 * len(idx)) // 10
        tr += idx[:k]
        te += idx[k:]
    tr.sort()
    te.sort()
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    mu = Xtr.mean(axis=0)
    sd = np.sqrt(((Xtr - mu) ** 2).sum(axis=0) / (len(Xtr) - 1))
    return Xtr, Xte, ytr, yte, (Xtr - mu) / sd, (Xte - mu) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_logistic(Z, y, eta, updates, record=()):
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Z @ w + b))
    for t in range(1, updates + 1):
        p = sigmoid(Z @ w + b)
        err = p - y
        w = w - eta * (Z.T @ err) / n
        b = b - eta * err.sum() / n
        if t in record:
            losses[t] = bce(y, sigmoid(Z @ w + b))
    return w, b, losses


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    acc = div(tp + tn, tp + fp + tn + fn)
    prec = div(tp, tp + fp)
    rec = div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec)
    spec = div(tn, tn + fp)
    return acc, prec, rec, f1, spec


def knn_prob(Zref, yref, Zq, K):
    """Share of positive neighbours among K nearest (stable tie-break by reference order)."""
    out = np.empty(len(Zq))
    for i, q in enumerate(Zq):
        d = ((Zref - q) ** 2).sum(axis=1)
        order = np.argsort(d, kind="stable")[:K]
        out[i] = yref[order].sum() / K
    return out


def knn_predict(Zref, yref, Zq, K):
    """Positive when at least (K+1)/2 neighbours are positive."""
    cnt = knn_prob(Zref, yref, Zq, K) * K
    return (np.rint(cnt) >= (K + 1) / 2).astype(int)
