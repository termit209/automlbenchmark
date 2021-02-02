import logging

import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qhoptim.pyt import QHAdam
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import lib 

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.results import save_predictions
from amlb.utils import Timer

from frameworks.shared.callee import save_metadata

log = logging.getLogger(__name__)

def run(dataset, config):
    log.info(f"\n**** NODE")

    is_classification = config.type == 'classification'

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc
    
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, n_jobs))
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))




    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]
    
    if is_classification:
        num_classes = len(set(y_train))
        ts = math.floor(time.time())
        experiment_name = f'node_adult_{ts}'
        model = nn.Sequential(lib.DenseBlock(num_features,layer_dim=128,num_layers=3,
                                         tree_dim=num_classes,flatten_output=False,depth=6,
                                         choice_function=lib.entmax15,bin_function=lib.entmoid15),
                          lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),).to(device)

        with torch.no_grad():
            res = model(torch.as_tensor(X_train[:2000], device=device))
        # trigger data-aware init
        trainer = lib.Trainer(model=model, loss_function=F.cross_entropy,experiment_name=experiment_name,
                          warm_start=False,Optimizer=QHAdam,optimizer_params=dict(nus=(0.7, 1.0),
                                                                                  betas=(0.95, 0.998)),
                          verbose=False,n_last_checkpoints=5)
        loss_history, err_history = [], []
        best_val_err = 10000000000
        best_step = 0
        early_stopping_rounds = 2500
        report_frequency = 1000

    for batch in lib.iterate_minibatches(X_train,
                                     y_train,
                                     batch_size=512, 
                                     shuffle=True,
                                     epochs=float('inf')):
        metrics = trainer.train_on_batch(*batch, device=device)
    
        loss_history.append(metrics['loss'])

        if trainer.step % report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')
            err = trainer.evaluate_classification_error(
            X_val,
            y_val,
            device=device,
            batch_size=128)
        
            if err < best_val_err:
                best_val_err = err
                best_step = trainer.step
                trainer.save_checkpoint(tag='best')
        
            err_history.append(err)
            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()

            print("Loss %.5f" % (metrics['loss']))
            print("Val Error Rate: %0.5f" % (err))
        
        if trainer.step > best_step + early_stopping_rounds:
            print('BREAK. There is no improvement for {} steps'.format(early_stopping_rounds))
            print("Best step: ", best_step)
            print("Best Val Error Rate: %0.5f" % (best_val_err))
            break

        

    with utils.Timer() as training:
        rf.fit(X_train, y_train)

    with utils.Timer() as predict:
        X_test = torch.as_tensor(X_val, device=device)
        with torch.no_grad():
            predictions = lib.process_in_chunks(trainer.model, X_test, batch_size=512)
        if is_classification:
            probabilities = F.softmax(prediction, dim=1)
        else:
            None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  training_duration=training.duration,
                  predict_duration=predict.duration)