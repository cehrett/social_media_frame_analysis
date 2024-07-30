import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, TraceMeanField_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, top_k_accuracy_score
from sklearn.metrics import roc_curve, auc
import argparse

# local code
from .utils.bayesian_account_clustering.data import ModelData, BatchIndices
from .utils.bayesian_account_clustering.model import AlphaNet, CoordinationModel
from .utils.bayesian_account_clustering import utils

def bayesian_clustering(needle_var, 
                        authors_path, 
                        successes_path, 
                        narratives_path, 
                        num_narratives, 
                        flags, 
                        component_weights, 
                        n_repeats, 
                        epochs, 
                        batch_size, 
                        num_particles, 
                        dropout_rate, 
                        cholesky_rank,
                        user_id='user_id'
                        ):
    pyro_version = '1.8.6'
    try:
        assert pyro.__version__.startswith(pyro_version), f"Pyro version mismatch. Expected: {pyro_version}, but found: {pyro.__version__}"
    except AssertionError as e:
        warnings.warn(str(e))

    paths = dict(
        authors=authors_path,
        successes=successes_path,
        narratives=narratives_path,
    )
    
    training_params = dict(
        epochs=epochs,
        batch_size=batch_size,
        num_particles=num_particles
    )

    # Get transformations of user-defined inputs
    num_flags = len(flags)
    num_components = len(component_weights)
    logprob_scale     = [0.5] + [1.8]*(num_components-1)

    ## Dataset names
    columns = dict(
        # column in authors csv containing author unique identifiers (e.g. 'handle')
        author_id_var = user_id,

        # column in authors csv containing ground truth account group identifiers. 
        # Set to None if no ground truth labels are available.
        needle_var = needle_var,

        # column in narratives and successes csv containing narrative identifiers (e.g. 'hashtags')
        narrative_var = 'cluster_labels',

        # column in narratives csv containing suspiciousness score (higher is more suspicious, e.g. 'kld')
        suspiciousness_var = 'kld',

        # column in authors csv containing number of trials for each account 
        # number of trials could be measured as number of days, messages, words, etc.
        trials_var = 'trials',

        # column in successes csv containing number of successess 
        # number of successes is measured as number of times the narrative appeared in the unit measured by trials
        successes_var = 'successes',
    )


    # adam optimizer settings
    initial_lr = 0.01
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    num_steps = training_params['epochs'] * 1e6/training_params["batch_size"]
    lrd = gamma ** (1 / num_steps) if num_steps>0 else 1
    adam_params = {"lr": initial_lr, "betas": (0.95, 0.999), "lrd": lrd}

    # tensor/device settings
    cuda = True
    device_num = 0 # only relevant for cuda=True
    dtype = torch.float32

    # validation
    assert num_components + num_flags > 0, 'must have at least 1 flag or narrative'

    if cuda:
        assert torch.cuda.is_available(), "Selected cuda=True, but cuda is not available."
        device = torch.device(f'cuda:{device_num}')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    # Data
    model_data = ModelData(paths, columns, num_components, num_narratives, flags, dtype, device)
    model_data.df_model.head()

    data = model_data.get_tensor_dict()
    # summarize the data object
    for k, v in data.items():
        print(k, v.shape, v.dtype, v.device)

    # get a test batch for visualizing the model
    test_batch = BatchIndices(data['trials'].shape[0], 12)[0]

    # See how much the needles use the suspicious clusters
    dd = model_data.df_model.copy()
    dd['sum_frames'] = dd[[col for col in dd.columns if isinstance(col,int)]].sum(axis=1)

    corr_df = dd[[col for col in dd if not isinstance(col,int) and col not in [user_id]]]
    print(corr_df.corr())

    # Set priors
    if num_narratives>0:
        # estimate narrative odds at account level
        p_ht = \
            data['successes'].sum(dim=0) / \
            data['trials'].sum(dim=0)

        p_ht_logodds = torch.log(p_ht / (1-p_ht))
        print("author: narrative log-odds:", p_ht_logodds)

    if num_flags>0:
        # estimate flag log odds
        p_flag = data['flags'].mean(dim=0)
        p_flag_logodds = torch.log(p_flag / (1-p_flag))
        print("author: flag log-odds:", p_flag_logodds)

    # prior for alpha
    p_alpha = torch.tensor(component_weights, device=device)
    p_alpha = p_alpha/p_alpha.sum()
    alpha_logit_loc = p_alpha.log()
    alpha_logit_prior = dist.Normal(loc=alpha_logit_loc, 
                                    scale = torch.tensor(logprob_scale, device=device)).to_event(1)
    print(f"prior p_alpha:\n\t{p_alpha}")
    print(f"alpha_logit_prior locs:\n\t{alpha_logit_loc}")

    locs = []
    scales = []
    if num_flags>0:
        # priors for flag reg coeff
        beta_loc = p_flag_logodds[:,None].expand(num_flags, num_components).clone().to(device)
        beta_loc[:,1:]=0.0 # non-majority users may be much more likely to have flag
        beta_scale = 3.*torch.ones(num_flags, num_components).to(device)
        beta_scale[:,0] = 0.3 # we're more confident about the majority group log odds

        locs.append(beta_loc)
        scales.append(beta_scale)

    # import pdb; pdb.set_trace()
    if num_narratives>0:
        # prior for narrative regression coefficients
        gamma_loc = p_ht_logodds[:,None].expand(num_narratives, num_components).clone().to(device)
        gamma_loc[:,1:] = 0.0  # non-majority users may be much more likely to use narrative
        gamma_scale = 3.*torch.ones(num_narratives, num_components).to(device)
        gamma_scale[:,0] = 0.3 # we're more confident about the majority group log odds

        locs.append(gamma_loc)
        scales.append(gamma_scale)

    beta_gamma_loc = torch.concat(locs, axis=0)
    beta_gamma_scale = torch.concat(scales, axis=0)
    beta_gamma_prior = dist.Normal(beta_gamma_loc.T, beta_gamma_scale.T).to_event(1)
    print(f"beta_gamma_prior:\n\t{beta_gamma_prior}")


    # Model

    def fit_model(supervised=False):

        # Encoder network
        alpha_net = AlphaNet(num_flags, num_narratives, num_components, dropout_rate=dropout_rate).to(device)

        # Probabilistic model
        cm = CoordinationModel(
            data, 
            device, 
            alpha_net, 
            alpha_logit_prior, 
            beta_gamma_prior,
            num_components,
            num_flags, 
            num_narratives,
            supervised=supervised,
            cholesky_rank = cholesky_rank
        )

        # Prepare for SVI
        pyro.clear_param_store()
        optimizer = ClippedAdam(adam_params)
        if supervised:
            loss = TraceMeanField_ELBO(num_particles=training_params['num_particles'])
        else:
            loss = TraceEnum_ELBO(num_particles=training_params['num_particles'])
        svi = SVI(config_enumerate(cm.model), cm.guide, optimizer, loss=loss)

        # Training loop
        dataset = BatchIndices(data['trials'].shape[0], training_params['batch_size'])
        tracking = []
        for epoch in range(training_params['epochs']):
            epoch_loss=0
            for step in range(len(dataset)):
                epoch_loss += svi.step(dataset[step])        

            print(f"Epoch {epoch+1} of {training_params['epochs']}. ELBO={-epoch_loss:0.1f}", end='\r')
            tracking.append({'epoch': epoch, 'step': step, 'loss': epoch_loss})

            # reshuffle dataset
            dataset.make_splits()
        print('\n')  # start a new line after finished training

        # Copy fit parameters
        pars = {}
        for k, v in pyro.get_param_store().items():
            pars[k] = v.detach().cpu()

        # Compute the confidence scores for each account from the alpha samples.
        alpha_logits = cm.compute_alpha_logodds(data)
        probs = torch.softmax(alpha_logits, dim=-1)

        # Build a run results object for analysis
        results = dict(
            encoder = alpha_net,
            model = cm,
            loss_curve = tracking,
            params = pars,
            p_clust = probs
        )

        return results


    # Unsupervised model
    ensemble = []
    for i in range(n_repeats):
        print(f"Fitting model {i+1} of {n_repeats}...")
        ensemble.append(fit_model())


    # Diagnostics
    if 'alphas_onehot' in data: # I.e. if we have needle data, we can do some diagnostics
        y_true = data['alphas_onehot'][:,1].cpu()
        # Set up the figure and subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Unsupervised loss curves
        for ix, m in enumerate(ensemble):
            df_tracking = pd.DataFrame(m['loss_curve'])
            axes[0].plot(df_tracking.epoch, df_tracking.loss, label=ix)
        axes[0].grid()
        axes[0].legend(loc=1, title='ensemble iter')
        axes[0].semilogy()
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Negative ELBO')
        axes[0].set_title('Unsupervised loss curves')

        # Precision-Recall curve
        p_not_cl0_avg = (sum([1-m['p_clust'][:,0] for m in ensemble]) / n_repeats).cpu()
        for ix, m in enumerate(ensemble):
            p_not_cl0 = (1-m['p_clust'][:,0]).cpu()
            prec, rec, _ = precision_recall_curve(y_true, p_not_cl0)
            axes[1].plot(rec, prec, color='blue', alpha=0.3, linestyle='dashed')
        prec, rec, _ = precision_recall_curve(y_true.cpu(), p_not_cl0_avg)
        avg_prec = average_precision_score(y_true.cpu(), p_not_cl0_avg)
        axes[1].plot(rec, prec, label=r'$\bar P=$' + f"{avg_prec:0.2f}", color='blue')
        axes[1].grid()
        axes[1].set_aspect('equal')
        axes[1].legend(loc=1)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Detection based on avg. prob.\nof not being in cluster 0')

        # ROC curves
        for ix, m in enumerate(ensemble):
            p_not_cl0 = (1 - m['p_clust'][:, 0]).cpu()
            fpr, tpr, _ = roc_curve(y_true, p_not_cl0)
            roc_auc = auc(fpr, tpr)
            axes[2].plot(fpr, tpr, color='black', alpha=0.3, linestyle='dashed', label=f'ROC curve of model {ix+1} (area = {roc_auc:.2f})')
        fpr_avg, tpr_avg, _ = roc_curve(y_true.cpu(), p_not_cl0_avg)
        roc_auc_avg = auc(fpr_avg, tpr_avg)
        axes[2].plot(fpr_avg, tpr_avg, color='blue', linewidth=3, label=f'Mean ROC (area = {roc_auc_avg:.2f})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].grid()
        axes[2].set_aspect('equal', adjustable='box')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title('Receiver Operating Characteristic')
        # axes[2].legend(loc="lower right")
        fig.tight_layout()

        # Save the figure
        plt.savefig(os.path.join('output', 'bayesian_clustering_results.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Account Clustering")
    
    parser.add_argument('--needle_var', type=str, default='in_US', help='Needle variable')
    parser.add_argument('--authors_path', type=str, required=True, help='Path to authors CSV file')
    parser.add_argument('--successes_path', type=str, required=True, help='Path to successes CSV file')
    parser.add_argument('--narratives_path', type=str, required=True, help='Path to narratives CSV file')
    parser.add_argument('--num_narratives', type=int, default=16, help='Number of narratives')
    parser.add_argument('--flags', nargs='+', default=['in_US'], help='List of flags')
    parser.add_argument('--component_weights', nargs='+', type=int, default=[10000, 10, 5, 1], help='Component weights')
    parser.add_argument('--n_repeats', type=int, default=40, help='Number of repeats')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_particles', type=int, default=1, help='Number of particles')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--cholesky_rank', type=int, default=2, help='Cholesky rank')

    args = parser.parse_args()
    
    bayesian_clustering(args.needle_var, args.authors_path, args.successes_path, args.narratives_path, args.num_narratives, args.flags, args.component_weights, args.n_repeats, args.epochs, args.batch_size, args.num_particles, args.dropout_rate, args.cholesky_rank)

