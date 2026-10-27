import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
SEED = 42

def load_data(data_dir=None):
    if data_dir is not None:
        import pandas as pd
        folder = Path(data_dir)
        def read(name):
            frame = pd.read_csv(folder / name, header=None, names=["label", "title", "description"], keep_default_na=False)
            return ((frame["title"] + " " + frame["description"]).tolist(), frame["label"].to_numpy(dtype=int) - 1)
        train, test = read("train.csv"), read("test.csv")
        if len(train[0]) != 120000 or len(test[0]) != 7600:
            raise ValueError("Expected 120000 training and 7600 test rows")
        return train, test
    from datasets import load_dataset
    ds = load_dataset('fancyzhx/ag_news')
    train, test = ds['train'], ds['test']
    if len(train) != 120000 or len(test) != 7600:
        raise ValueError('Unexpected AG News split sizes')
    return (list(train['text']), np.asarray(train['label'])), (list(test['text']), np.asarray(test['label']))

def model():
    return MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                         learning_rate_init=0.001, batch_size=256, max_iter=15,
                         early_stopping=False, random_state=SEED, tol=1e-4)

def stratified_indices(y, count=200):
    rng = np.random.default_rng(SEED)
    if count % 4: raise ValueError('Count must be divisible by 4')
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == i), count // 4, replace=False)
                                   for i in range(4)]))

def run(outdir, data_dir=None):
    from joblib import dump
    outdir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_data(data_dir)
    print('Train:', len(x_train), 'Test:', len(x_test))
    for i, name in enumerate(LABELS):
        print(f'{name}: train={(y_train == i).sum()}, test={(y_test == i).sum()}')
        print('Example:', x_train[int(np.flatnonzero(y_train == i)[0])][:500])
    results = []
    (outdir / 'dataset_summary.json').write_text(json.dumps({
        'training_count': len(x_train), 'test_count': len(x_test),
        'categories': {name: {'train': int((y_train == i).sum()), 'test': int((y_test == i).sum()),
                             'example': x_train[int(np.flatnonzero(y_train == i)[0])]}
                       for i, name in enumerate(LABELS)}}, indent=2), encoding='utf-8')
    best = (-1, None, None, None)
    for kind in ('CountVectorizer', 'TfidfVectorizer'):
        for stop in (False, True):
            vectorizer = (CountVectorizer if kind == 'CountVectorizer' else TfidfVectorizer)(
                stop_words='english' if stop else None, max_features=20000, dtype=np.float32)
            X = vectorizer.fit_transform(x_train)
            T = vectorizer.transform(x_test)
            sparsity = 100 * (1 - X.nnz / (X.shape[0] * X.shape[1]))
            clf = model()
            started = time.perf_counter()
            clf.fit(X, y_train)
            pred = clf.predict(T)
            elapsed = time.perf_counter() - started
            acc = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='macro', zero_division=0)
            key = f'{kind}_stop_{stop}'
            record = dict(configuration=key, accuracy=acc, macro_precision=precision,
                          train_shape=list(X.shape), sparsity_percent=sparsity,
                          epochs=clf.n_iter_, training_seconds=elapsed,
                          confusion_matrix=confusion_matrix(y_test, pred, labels=range(4)).tolist())
            results.append(record)
            print(json.dumps(record, indent=2))
            if acc > best[0]: best = (acc, vectorizer, clf, pred)
            if kind == 'CountVectorizer' and not stop:
                (outdir / 'count_dtm_summary.json').write_text(json.dumps({
                    'shape': list(X.shape), 'nonzero_entries': int(X.nnz),
                    'sparsity_percent': sparsity}, indent=2), encoding='utf-8')
    _, vec, clf, predictions = best
    dump(vec, outdir / 'best_vectorizer.joblib')
    dump(clf, outdir / 'best_mlp.joblib')
    sample = stratified_indices(y_test)
    with (outdir / 'comparison_sample.json').open('w', encoding='utf-8') as f:
        json.dump([dict(index=int(i), text=x_test[i], true_category=LABELS[y_test[i]],
                        mlp_category=LABELS[int(predictions[i])]) for i in sample], f, indent=2)
    errors = np.flatnonzero(predictions != y_test)[:3]
    with (outdir / 'misclassified.json').open('w', encoding='utf-8') as f:
        json.dump([dict(text=x_test[i], true_category=LABELS[y_test[i]],
                        predicted_category=LABELS[int(predictions[i])]) for i in errors], f, indent=2)
    (outdir / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (outdir / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['configuration', 'accuracy', 'macro_precision', 'sparsity_percent', 'epochs', 'training_seconds'])
        w.writeheader()
        for r in results: w.writerow({k: r[k] for k in w.fieldnames})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs')
    p.add_argument('--data-dir', default=None, help='Directory containing train.csv and test.csv')
    args = p.parse_args()
    run(Path(args.output), args.data_dir)
