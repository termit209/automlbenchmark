import logging

import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qhoptim.pyt import QHAdam
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from . import lib
import math
import time
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, log_loss

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.results import save_predictions
from amlb.utils import Timer

from frameworks.shared.callee import save_metadata, result

log = logging.getLogger(__name__)

def run(dataset, config):
    log.info(f"\n**** NODE")

    is_classification = config.type == 'classification'
    
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc
    print(X_train.dtype, y_train.dtype)
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    #log.info("Running RandomForest with a maximum time of {}s on {} cores.".format(config.max_runtime_seconds, n_jobs))
    #log.warning("We completely ignore the requirement to stay within the time limit.")
    #log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))
    print(config)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]
    ts = math.floor(time.time())
    experiment_name = f'node_adult_{ts}'
    if is_classification:
        X_test = X_test.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.long)
        y_test = y_test.astype(np.long)
        y_val = y_val.astype(np.long)
        num_classes = len(set(y_train))

        model = nn.Sequential(lib.DenseBlock(num_features,layer_dim=128,num_layers=3,
                                         tree_dim=num_classes,flatten_output=False,depth=6,
                                         choice_function=lib.entmax15,bin_function=lib.entmoid15),
                          lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),).to(device)


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
                err = trainer.evaluate_classification_error(X_val,y_val,device=device,batch_size=128)
        
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
    else:
        X_test = X_test.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        y_val = y_val.astype(np.float32)
        model = nn.Sequential(lib.DenseBlock(num_features, 2048, num_layers=1, tree_dim=3, depth=6, flatten_output=False,
        choice_function=lib.entmax15, bin_function=lib.entmoid15), lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),).to(device)

        optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }

        trainer = lib.Trainer(model=model, loss_function=F.mse_loss,experiment_name=experiment_name,
                              warm_start=False,Optimizer=QHAdam,optimizer_params=optimizer_params,
                              verbose=False,n_last_checkpoints=5)


        loss_history, mse_history = [], []
        best_mse = float('inf')
        best_step_mse = 0
        early_stopping_rounds = 5000
        report_frequency = 100


        for batch in lib.iterate_minibatches(X_train, y_train, batch_size=512, 
                                             shuffle=True, epochs=float('inf')):
            metrics = trainer.train_on_batch(*batch, device=device)
    
            loss_history.append(metrics['loss'])

            if trainer.step % report_frequency == 0:
                trainer.save_checkpoint()
                trainer.average_checkpoints(out_tag='avg')
                trainer.load_checkpoint(tag='avg')
                mse = trainer.evaluate_mse(X_valid, y_valid, device=device, batch_size=1024)

                if mse < best_mse:
                    best_mse = mse
                    best_step_mse = trainer.step
                    trainer.save_checkpoint(tag='best_mse')
                mse_history.append(mse)
        
                trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()

                print("Loss %.5f" % (metrics['loss']))
                print("Val MSE: %0.5f" % (mse))
            if trainer.step > best_step_mse + early_stopping_rounds:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step_mse)
                print("Best Val MSE: %0.5f" % (best_mse))
                break
        


    with Timer() as predict:
        X_test = torch.as_tensor(X_test, device=device)
        with torch.no_grad():
            output = lib.process_in_chunks(trainer.model, X_test, batch_size=512)
        
        if is_classification:
            probabilities = F.softmax(output, dim=1)
            probabilities = probabilities.cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            print(classification_report(y_test, predictions))
            if probabilities.shape[1] == 2:
                print(roc_auc_score(y_test, probabilities[:,1]))
            else:
                print(log_loss(y_test, probabilities))
        else:
            predictions = output.cpu().numpy()
        
   

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=True)
                     
    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=True,
                  predict_duration=predict.duration)
