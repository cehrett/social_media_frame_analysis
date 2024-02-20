import pandas as pd
import torch
import torch.nn.functional as F

class ModelData:
    def __init__(self, paths, columns, num_components, num_narratives, flags, dtype, device, needle_2_id = None):
        self.paths = paths
        self.columns = columns
        self.num_components = num_components
        self.num_narratives = num_narratives
        self.flags = flags
        self.dtype = dtype
        self.device = device
        self.needle_2_id = needle_2_id
        
        # load author data. Steps:
        # 1. read in data
        # 2. make sure there's exactly 1 row per author
        self.df_author = pd.read_csv(self.paths['authors'])
        # import pdb; pdb.set_trace()
        assert self.df_author.groupby([self.columns['author_id_var']]).size().max() == 1, 'Can only have 1 row per author identifier.'
        
        # load narratives data. Steps:
        # 1. read in data and sort by decreasing suspiciousness (KL-divergence)
        # 2. make sure there's exactly 1 row per narrative
        # 3. Save a list of the top <num_narratives> narratives in order of decreasing suspiciousness
        self.df_narratives = \
            pd.read_csv(self.paths['narratives']).\
            sort_values(self.columns['suspiciousness_var'], ascending=False)
        assert self.df_narratives.value_counts(self.columns['narrative_var']).max() == 1, \
            "Must have exactly 1 of each narrative"
        self.top_narratives = self.df_narratives[self.columns['narrative_var']][:self.num_narratives].tolist()
        
        # load successes data and create model table. Steps: 
        # 1. read in data
        # 2. make sure there's only one row per author-narrative pair
        # 3. filter to narratives in top_narratives list
        # 4. reshape so each hastag gets its own column. Note: some authors do not mention any of the top narratives, so they are dropped
        # 5. restore missing authors by merging on author table and imputing zeros to all success counts. Note: this also brings over flags and trials fields
        # 6. ensure that the author_id_var list matches the original author table
        self.df_successes = pd.read_csv(self.paths["successes"])
        assert self.df_successes.value_counts([self.columns["author_id_var"], self.columns["narrative_var"]]).max() == 1, \
            "Must have exactly 1 of each author-narrative pair"

        self.df_model = \
            self.df_successes.query(f'{self.columns["narrative_var"]} in @self.top_narratives').\
            pivot_table(index=self.columns['author_id_var'], columns=self.columns["narrative_var"], values=self.columns["successes_var"], fill_value=0).\
            reset_index().\
            merge(self.df_author, how='right').\
            fillna(0)
        
        print(self.df_model.shape)

        assert self.df_model[self.columns['author_id_var']].equals(self.df_author[self.columns['author_id_var']]), \
            "The successes author list does not match the original author list. This will probably end badly if not fixed."
        
    def get_tensor_dict(self):
        # convert model data into pytorch tensors 
        data = dict()
        data['trials'] = torch.tensor(self.df_model[self.columns["trials_var"]].values, dtype=self.dtype).to(self.device)[:,None]
        data['successes'] = torch.tensor(self.df_model.loc[:, self.top_narratives].values, dtype=self.dtype).to(self.device)
        data['flags'] = torch.tensor(self.df_model.loc[:, self.flags].values, dtype=self.dtype).to(self.device)
        
        # if needle variable is provided add to data
        if self.columns["needle_var"] in self.df_model:
            needles = self.df_model[self.columns["needle_var"]]
            if self.needle_2_id is not None:
                # map levels to integers
                needles = needles.map(self.needle_2_id)
            
            # create the data object
            data['alphas'] = torch.tensor(needles.values, dtype=self.dtype).to(self.device)
            alphas_onehot = torch.zeros(data['trials'].shape[0], self.num_components)
            num_codes = len(data['alphas'].unique())
            alphas_onehot[:, :num_codes] = F.one_hot(data['alphas'].type(torch.long))
            data['alphas_onehot'] = alphas_onehot.to(self.device)
            
        # cap successes at trials if more successes than trials
        if self.num_narratives>0:
            data['successes'] = torch.where(
                data['trials']  < data['successes'],
                data['trials'],
                data['successes'] 
            )
            
        return data
    
# this class helps load batches of data 
# for the variational inference optimization
class BatchIndices():
    def __init__(self, num_authors, batch_size):
        self.num_authors = num_authors
        self.batch_size = batch_size
        self.make_splits()
        
    def __len__(self):
        return len(self.splits)
    
    def __getitem__(self, ix):
        return self.splits[ix]
    
    def make_splits(self):
        """
        Reshuffle the indices and give a new set of batch splits.
        Recommended to do this each training epoch.
        """
        indices = torch.randperm(self.num_authors)
        self.splits = torch.split(indices,self.batch_size)
        