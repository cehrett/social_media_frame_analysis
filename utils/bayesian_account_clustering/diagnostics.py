import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyro.distributions import Categorical
import pyro
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot as plt

# font_path = "/zfs/disinfo/narratives/fonts/NotoSansCJKtc-Regular.otf"
# fprop = matplotlib.font_manager.FontProperties(fname=font_path, size=14)
# font = {'size'   : 18}
# matplotlib.rc('font', **font)
# matplotlib.rc('lines', linewidth=2.0)
# matplotlib.rc("errorbar", capsize=10.0)

MISSING_ACCOUNT_LABELS_MSG = 'This function requires ground truth account labels.'

class ModelDiagnostics():
    """
    This class provides model diagnostic tools given provided posterior samples.
    """
    
    def __init__(self, flag_features, narrative_features, trials, params, group_probs, account_labels = None):

        self.flag_features = flag_features
        self.narrative_features = narrative_features
        self.trials = trials
        self.params = params
        self.group_probs = group_probs
        self.account_labels = account_labels
        
        # extract shapes
        self.num_components = self.group_probs.shape[-1]
        self.num_accounts = self.flag_features.shape[0]
       
    
    def get_cluster_stats(self, 
                          prior_shares = None,
                          num_top_hashtags=5):
        """
        Compute some basic summary statistics for each cluster.
        """
        num = self.group_probs.sum(axis=0).numpy()
        share = num / self.num_accounts 
        df = pd.DataFrame({
            'exp_num_accounts': num, 
            'exp_share_accounts': share,
        }) 
        df = df.reset_index().rename(columns={'index': 'cluster_id'})
        
        if prior_shares is not None:
            df['prior_shares'] = prior_shares
        
        # compute homogeneity for each cluster as 
        # the average L2 distance to the mean for
        # the top N accounts in the cluster. where
        # N is the expected number of accounts and 
        top_ix = self.group_probs.argsort(descending=True, dim=0)
        homog_flag = []
        homog_narr = []
        for ix, n in enumerate(num): 
            n = int(n.item()) 
            if self.flag_features.shape[-1] > 0:
                x_flag = self.flag_features[top_ix[:,ix]][:n]
                homog_flag.append(self.get_homogeneity(x_flag).item())
            if self.narrative_features.shape[-1] > 0:
                x_narr = self.narrative_features[top_ix[:,ix]][:n]
                homog_narr.append(self.get_homogeneity(x_narr).item())
            
        df['homog_flag'] = homog_flag
        df['homog_narr'] = homog_narr
        
        return df
    
    @staticmethod
    def get_homogeneity(x, normalize=True):
        """
        Compute the average L2 distance to the mean. 
        Optionally normalize by the L2 length of the mean vector
        """
        mu = x.mean(axis=0)
        homog = torch.sqrt(((x - mu)**2).mean())
        if normalize:
            homog = homog / torch.sqrt((mu**2).mean())
            
        return  homog
    
    def get_cluster_accounts(self, df_authors, cluster_id, author_id_cols = ['handle'], username_col = 'handle', num_accounts=100):
        """
        Return a list of accounts associated with the provided cluster_id sorted in 
        decreasing confidence of cluster membership.
        """
        assert username_col in author_id_cols, f'username_col {username_col} must be in author_id_cols {author_id_cols}'
        
        dat = df_authors
        alpha_conf_ix = self.group_probs[:,cluster_id]
        dat[f'alpha_conf_{cluster_id}'] = alpha_conf_ix
        dat = dat.sort_values(f'alpha_conf_{cluster_id}', ascending=False).iloc[:num_accounts]
        cols = author_id_cols + [f'alpha_conf_{cluster_id}']
        
        # make clickable link to user profile
        if username_col is not None:
            def make_clickable(handle):
                return f'<a target="_blank" href="https://twitter.com/{handle}">{handle}</a>'
            dat = dat.style.format({username_col: make_clickable})
        else:
            dat = dat[cols]
        
        return dat
    
    def plot_detection_curve(self, target_class_id):
        targets = torch.isin(self.account_labels, target_class_id).numpy().astype(int)
        
        fig = plt.gcf()
        fig.set_size_inches(8,8)
        ax = plt.gca()
        for i in range(self.num_components):
            prec, rec, eps = precision_recall_curve(targets, self.group_probs[:,i])
            avg_prec = average_precision_score(targets, self.group_probs[:,i])
            plt.plot(rec, prec, 'o-', label=f"clust-id {i}, targ-id: {target_class_id.item()}: P={avg_prec:0.5f}")

        ax.set_aspect('equal')
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Account Detection Curves')
        plt.legend()
        
        return plt
            
    def detect_label_assignments(self, cost_metric_fn = metrics.average_precision_score):
        """
        This function attempts to match the model components to the account
        labels provided in true_values. Auroc is computed for each possible
        assignment and the hungarian algorithm is used to compute the best 
        linear sum assignment. If pred_values are not supplied, they are 
        computed from the alpha samples.
        
        Return:
            A dictionary with keys corresponding to the unique true_values and 
            values corresponding to cluster_id.
        """
        assert self.account_labels is not None, \
            MISSING_ACCOUNT_LABELS_MSG
        
        # use auroc to compute cost matrix
        unique_labels = self.account_labels.unique().cpu().numpy()
        cost_matrix = np.zeros((len(unique_labels), self.num_components))
        for lab_ix, lab in enumerate(unique_labels):
            true = (self.account_labels==lab).type(torch.int).cpu().numpy()
            for k in range(self.num_components):
                pred = self.group_probs[:,k]
                cost_matrix[lab_ix, k] = cost_metric_fn(true, pred)
                
        # find best assignments
        rows, cols = linear_sum_assignment(cost_matrix, maximize=True)
        
        mapping = {rows[i]: cols[i] for i, _ in enumerate(unique_labels)}
                
        return mapping
    
    def compute_detection_stats(self, label_assignments = None, label_names = None):
        """
        Assess the correspondence between the clustering results and the provided true account labels.
        """
        assert self.account_labels is not None, \
            MISSING_ACCOUNT_LABELS_MSG
        
        if label_assignments is None:
            label_assignments = self.detect_label_assignments(true_values)
            
        unique_labels = self.account_labels.unique().cpu().numpy()
        alpha_confs = self.group_probs.cpu().numpy()
        results = []
        for lab_ix, lab in enumerate(label_assignments.keys()):
            true = (self.account_labels==lab).type(torch.int).cpu().numpy()
            pred = alpha_confs[:, label_assignments[lab]]
            pred_thresh = (np.argmax(alpha_confs, axis=-1)==label_assignments[lab]).astype(int)
            
            results.append(dict(
                true_label = lab if label_names is None else label_names[lab],
                cluster_id = label_assignments[lab],
                auroc = metrics.roc_auc_score(true, pred),
                avg_precision = metrics.average_precision_score(true, pred),
                f1 = metrics.f1_score(true, pred_thresh), 
                balanced_accuracy = metrics.balanced_accuracy_score(true, pred_thresh)
            ))
            
        return pd.DataFrame(results)
    
    @staticmethod
    def get_quantiles(dist, low=0.05, med=0.5, high=0.95, probs=True):
        qlow = dist.icdf(torch.tensor(low))
        qmed = dist.icdf(torch.tensor(med))
        qhigh = dist.icdf(torch.tensor(high))

        if probs:
            qlow, qmed, qhigh = [torch.sigmoid(q) for q in [qlow, qmed, qhigh]]

        return qlow, qmed, qhigh

    def get_posterior_coeffs(self, dist, feature_labels, probs=True, top_n=None):
        low, med, high = self.get_quantiles(dist, probs=probs)
        low, med, high = [t.numpy() for t in (low, med, high)]
        dfs = []
        for ix in range(low.shape[0]):
            df = pd.DataFrame({ 'cluster': ix, 'variable': np.array(feature_labels), 'low': low[ix], 'med': med[ix], 'high': high[ix]})  
            df = df.sort_values(['low'], ascending=False)
            if top_n is not None:
                df = df.head(top_n)
            
            dfs.append(df.sort_values(['med'], ascending=False))
        
        return pd.concat(dfs, axis=0, ignore_index=True)     
    
    def plot_posterior_coeffs(self, beta_dist, gamma_dist, sort_cluster, top_n = None,
                              label_map = None, flag_labels=None, narrative_labels=None, plot_probs=False, label=None, fig_ax = None, sort_ix = None):
        """
        label_map is of the form {cluster_id: axis_id}.
        only mapped clusters will be plotted
        """
        if label_map is None:
            label_map = {i:i for i in range(self.num_components)}
            
        if top_n is None:
            top_n = len(narrative_labels)
            
        config_dict = {}
            
        if beta_dist is not None:
            beta_low, beta_mid, beta_high = self.get_quantiles(beta_dist, probs = plot_probs)
            beta_low, beta_mid, beta_high = [t.t() for t in (beta_low, beta_mid, beta_high)]
            
            if sort_cluster is not None:
                if sort_ix is None:
                    sort_ix_beta = torch.argsort(beta_mid[...,sort_cluster])[-top_n:]
                else:
                    sort_ix_beta = sort_ix[0][-top_n:]
                beta_low = beta_low[sort_ix_beta]
                beta_mid = beta_mid[sort_ix_beta]
                beta_high = beta_high[sort_ix_beta]
                config_dict['sort_ix_beta'] = sort_ix_beta
                
            beta_err = torch.stack([beta_mid - beta_low, beta_high-beta_mid])
            
            num_flags = beta_dist.shape()[-1]
        else:
            num_flags=0
            
        if gamma_dist is not None:
            gamma_low, gamma_mid, gamma_high = self.get_quantiles(gamma_dist, probs = plot_probs)
            gamma_low, gamma_mid, gamma_high = [t.t() for t in (gamma_low, gamma_mid, gamma_high)]
            
            if sort_cluster is not None:
                if sort_ix is None:
                    sort_ix_gamma = torch.argsort(gamma_mid[...,sort_cluster])[-top_n:]
                else:
                    sort_ix_gamma = sort_ix[1][-top_n:]
                gamma_low = gamma_low[sort_ix_gamma]
                gamma_mid = gamma_mid[sort_ix_gamma]
                gamma_high = gamma_high[sort_ix_gamma]
                config_dict['sort_ix_gamma'] = sort_ix_gamma
                
            gamma_err = torch.stack([gamma_mid - gamma_low, gamma_high-gamma_mid]) 
        
        # make the plot
        if fig_ax is None:
            fig, axs = plt.subplots(ncols=self.num_components, nrows=2, sharey=False, sharex=True,
                                    gridspec_kw={'height_ratios': [top_n, num_flags]})
            fig.set_size_inches(0.5*15*self.num_components/3, max(5, 0.5*(num_flags + top_n)/2))
            fig.set_facecolor('white')
            eps=0
        else:
            fig, axs = fig_ax
            eps=0.15

        # plot gamma intervals
        if top_n>0:
            ys = np.arange(top_n)+eps
            for ix in range(self.num_components):
                if ix in label_map.keys():
                    axix = label_map[ix]
                    axs[0, axix].errorbar(x=gamma_mid[...,ix], y=ys, xerr=gamma_err[...,ix], label=label, 
                                        fmt='.', capthick=2, capsize=5)
                    axs[0, axix].set_yticks(ys, ['']*top_n)
                    axs[0, axix].set_title(f"Clust. {label_map[ix]}")
                if ix==0 and narrative_labels is not None:
                    axs[0, ix].set_yticklabels(np.array(narrative_labels)[sort_ix_gamma], fontproperties=fprop)
                    axs[0, ix].legend(loc=4, fontsize=14)
                
        # plot beta intervals
        if num_flags>0:
            ys = np.arange(num_flags)+eps
            for ix in range(self.num_components):
                if ix in label_map.keys():
                    axix = label_map[ix]
                    axs[1, axix].errorbar(x=beta_mid[...,ix], y=ys, xerr=beta_err[...,ix], label=label, 
                                        fmt='.', capthick=2, capsize=5)
                    axs[1, axix].set_yticks(ys, ['']*num_flags)
                if ix==0 and flag_labels is not None:
                    axs[1, ix].set_yticklabels(np.array(flag_labels)[sort_ix_beta], fontproperties=fprop)
                
        if fig_ax is None:
            for ax in axs.flatten():
                ax.grid()
        
        fig.tight_layout()
        
        return (fig, axs), (sort_ix_beta, sort_ix_gamma)
