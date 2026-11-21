import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision_tree import DecisionTree
from src.preprocessing import TASKS, Preprocessor, get_split, load_task
from src.random_forest import RandomForest

RESULTS = Path('results')


def make_model(model, params):
    cls = DecisionTree if model == 'tree' else RandomForest
    return cls(**params)



def candidates(model, seed, fast, task):
    if model == 'tree':
        if task == 'image':
            depths = (40,)
            leaves = (1, 3, 5)
        else:
            depths = (4, 6, 8, 10)
            leaves = (2, 10)

        for depth in depths:
            for leaf in leaves:
                yield {
                    'max_depth': depth,
                    'min_samples_leaf': leaf,
                    'max_thresholds': 64 if fast else None,
                    'random_state': seed,
                }
    else:
        yield {
            'n_estimators': 100,
            'max_depth': 25,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_thresholds': 32,
            'random_state': seed,
        }



def predict_with_voting(model, X, model_type, voting):
    if model_type == 'forest' and voting == 'soft':
        return model.classes_[np.argmax(model.predict_proba(X), axis=1)]
    return model.predict(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=[*TASKS, 'all'], default='all')
    parser.add_argument('--model', choices=['tree', 'forest'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast', action='store_true', help='Smaller, approximate exploratory sweep')
    parser.add_argument('--final', action='store_true', help='Fit chosen configuration on full training data')
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    # Run the same workflow for each requested dataset.
    for task in TASKS if args.task == 'all' else [args.task]:
        X, y, X_test, ids = load_task(task)
        best_path = RESULTS / f'{args.model}_{task}_best.json'

        # Refit the selected configuration on all labeled data before predicting test labels.
        if args.final:
            if not best_path.exists():
                raise FileNotFoundError(f'Run validation first: {best_path}')
            best = json.loads(best_path.read_text())
            params = best['params']
            voting = best.get('voting', 'hard')
            prep = Preprocessor('none').fit(X)
            model = make_model(args.model, params).fit(prep.transform(X), y)
            pred = predict_with_voting(model, prep.transform(X_test), args.model, voting)
            output = RESULTS / f'{args.model}_{task}_test_predictions.csv'
            pd.DataFrame({'id': ids, 'label': pred}).to_csv(output, index=False)
            print(f'{task}: wrote {len(pred)} {voting}-voting test predictions to {output}')
            continue

        # Use the shared saved split so all models are evaluated on identical examples.
        train, val = get_split(task, y)
        prep = Preprocessor('none').fit(X[train])
        X_train = prep.transform(X[train])
        X_val = prep.transform(X[val])

        rows = []
        best_acc = -1.0
        best_params = None
        best_pred = None
        best_voting = None

        # Train each candidate and compare validation accuracy.
        for params in candidates(args.model, args.seed, args.fast, task):
            start = time.perf_counter()
            model = make_model(args.model, params).fit(X_train, y[train])
            train_pred = model.predict(X_train)
            hard_pred = model.predict(X_val)
            train_accuracy = float(np.mean(train_pred == y[train]))
            hard_accuracy = float(np.mean(hard_pred == y[val]))

            soft_accuracy = None
            voting = 'hard'
            pred = hard_pred
            accuracy = hard_accuracy
            # Evaluate both voting rules and keep the better one for this forest.
            if args.model == 'forest':
                soft_pred = predict_with_voting(model, X_val, args.model, 'soft')
                soft_accuracy = float(np.mean(soft_pred == y[val]))
                if soft_accuracy > hard_accuracy:
                    voting = 'soft'
                    pred = soft_pred
                    accuracy = soft_accuracy
                print(f'  Hard voting: {hard_accuracy:.4f} | '
                      f'Soft voting: {soft_accuracy:.4f} | Selected: {voting}', flush=True)

            elapsed = time.perf_counter() - start
            rows.append({
                'params': json.dumps(params),
                'train_acc_hard': train_accuracy,
                'val_acc_hard': hard_accuracy,
                'val_acc_soft': soft_accuracy,
                'val_acc': accuracy,
                'voting': voting,
                'seconds': round(elapsed, 2),
            })
            print(f'{task} {args.model}: train_hard={train_accuracy:.4f} '
                  f'val_selected={accuracy:.4f} voting={voting} '
                  f'time={elapsed:.1f}s {params}', flush=True)

            # Track the best configuration and its validation predictions.
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = params
                best_pred = pred.copy()
                best_voting = voting

        pd.DataFrame(rows).to_csv(RESULTS / f'{args.model}_{task}_sweep.csv', index=False)
        best_path.write_text(json.dumps({
            'params': best_params,
            'val_acc': best_acc,
            'voting': best_voting,
        }, indent=2))
        pd.DataFrame({'row': val, 'true': y[val], 'pred': best_pred}).to_csv(
            RESULTS / f'{args.model}_{task}_val_predictions.csv', index=False)
        print(f'Best {task} {args.model}: {best_acc:.4f} ({best_voting} voting)')


if __name__ == '__main__':
    main()
