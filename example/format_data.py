import pandas as pd
import numpy as np

DATA_FOLDER = "/Users/margaretli/gitfiles/scaling-app-copy/example"

def precise_flops_per_token_chinchilla(width, depth):
    seq_len = 2048
    vocab_size = 50432
    num_heads = 4
    width = width.astype(float)
    depth = depth.astype(float)

    embeddings = 2 * seq_len * width

    attention = 2 * 3 * seq_len * (width ** 2)
    kq_logits = 2 * seq_len * seq_len * width
    softmax = 3 * num_heads * seq_len * seq_len
    softmax_q_red = 2 * seq_len * seq_len * width
    final_linear = 2 * seq_len * (width ** 2)
    attention += kq_logits + softmax + softmax_q_red + final_linear

    ffw_size = 4 * width # check this, in the paper it is 4 * width
    dense_block = 4 * seq_len * width * ffw_size
    final_logits = 2 * seq_len * width * vocab_size
    forward_pass = embeddings + depth * attention + depth * dense_block + final_logits
    backward_pass = 2 * forward_pass
    return (forward_pass + backward_pass) / seq_len

def precise_param_count_open_lm(width, depth, vocab_size=50432):
    d_ff = 256 * (((2 * 4 * width / 3).astype(int) + 256 - 1) // 256)
    return (4 * width + 3 * d_ff) * width * depth + vocab_size * width

def apply_smoothing_filter(df, filter_func, compensate_for_logging_delay=True, key='train/loss', **filter_args):
    out = []
    for _, row in df.iterrows():
        if len(row[key]) == 0:
            out.append(None)
            continue
        filtered = filter_func(row[key].dropna(), **filter_args)
        if compensate_for_logging_delay:
            filtered.index = filtered.index - np.diff(filtered.index, prepend=0)/2
        out.append(filtered)
    return out

def proportional_sliding_window_filter(x, p=0.05):
    # assert that the index of x has constant increments?
    x_cumsum = x.cumsum().values
    x_cumsum_pad = np.concatenate([[0], x_cumsum])
    inds = np.arange(len(x))
    inds_up = np.minimum(inds + np.floor(p * inds).astype(int), len(x)-1)
    inds_down = np.maximum(0, inds - np.floor(p * inds).astype(int))
    inds_new = (inds_up + inds_down)/2
    index_new = np.interp(inds_new, inds, x.index)
    try:
        x_series = pd.Series((x_cumsum[inds_up] - x_cumsum_pad[inds_down]) / (inds_up - inds_down+1),
                     index=index_new, name=x.name + '_smoothed')
    except:
        x_series = pd.Series((x_cumsum[inds_up] - x_cumsum_pad[inds_down]) / (inds_up - inds_down+1),
                     index=index_new, name="" + '_smoothed')
    return x_series


def add_columns_to_rsld_df(df):
    """ Calculate C, N, etc., for Porian, e.t. al data
    """
    df = df.copy()

    # Counting parameters
    df['params_active'] = (12 * (df.width**2) * df.depth + df.vocab_size * df.width).astype(float)
    df['params_active_precise'] = precise_param_count_open_lm(df.width, df.depth)
    df['params_no_embed'] = precise_param_count_open_lm(df.width, df.depth, vocab_size=0)
    df['params_all'] = 12 * (df.width**2) * df.depth + (df.seq_len + 2 * df.vocab_size) * df.width

    # Counting FLOPs
    df['flops_per_token_att_no_embed'] = 6 * df['params_no_embed'] + 6 * df.seq_len * df.width * df.depth
    df['flops_per_token_att'] = 6 * df['params_active_precise']  + 6 * df.seq_len * df.width * df.depth
    df['flops_per_token_cc'] = precise_flops_per_token_chinchilla(df['width'], df['depth'])
    df['flops_per_token_no_att'] = 6 * df['params_active_precise']
    df['flops_per_token_no_att_no_embed'] = 6 * df['params_no_embed']
    df['flops_per_token'] = df['flops_per_token_no_att']

    df['params'] = df['flops_per_token'] / 6
    df['eff_params_att'] = df['flops_per_token_att'] / 6


    df['train/loss_smoothed'] = apply_smoothing_filter(df, proportional_sliding_window_filter, compensate_for_logging_delay=True, key='train/loss')
    for k in df:
        if k.startswith('train/') and k.endswith('_loss'):
            df[k + '_smoothed'] = apply_smoothing_filter(df, proportional_sliding_window_filter, compensate_for_logging_delay=False, key=k)
    return df

def get_rsld_data(config_name, data_file=f"{DATA_FOLDER}/rsld/experiment_results.pickle.xz", ckpt=False):
    """
    Get raw data from Porian, et al
    """

    ISOFLOP_ARGS = {
        ('kaplan', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token_no_att_no_embed', n_key='params_no_embed'),
        ('standard', 'val'):  dict(loss_key='val/loss', flop_per_token_key='flops_per_token', n_key='params'),
        ('standard', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token', n_key='params'),
        ('attention', 'train'): dict(loss_key='train/loss_smoothed', flop_per_token_key='flops_per_token_att', n_key='eff_params_att'),
    }

    NAME_TO_CONFIG_DICT = {
        'rsld': ('rw', 'base', 'short', 'chinchilla', 'standard', 'train'), #: 'Cosine decay', # original Chinchilla?
        'misfitting': ('c4', 'misfitting', 'misfitting', 'cosine', 'standard', 'train'), # ours
    }

    df = pd.read_pickle(data_file, compression='xz')
    df = add_columns_to_rsld_df(df)
    # select only the rows for a particular config
    config = NAME_TO_CONFIG_DICT[config_name]
    dataset, hparams, warmup, decay, param_count, val = config
    isoflop_args = ISOFLOP_ARGS[config[-2:]]
    df = df.query(f"dataset=='{dataset}' and hparams=='{hparams}' and warmup=='{warmup}' and decay=='{decay}'").copy()

    # get C, D, loss at last checkpoint
    flops_field_name = isoflop_args['flop_per_token_key']
    loss_key = isoflop_args['loss_key']
    Ds, current_steps, portion_steps_elapsed = [], [], []
    for ind, row in df.iterrows():
        df.loc[ind, 'total_steps'] = row[loss_key].index[-1]
        current_steps.append(row[loss_key].index.astype(float).to_list())
        portion_steps_elapsed.append((row[loss_key].index.astype(float)/row[loss_key].index[-1]).to_list())
        Ds.append(np.array((row[loss_key].index.astype(float) * row['seq_len'] * row['bs']).to_list()))
        row[loss_key].index = row[loss_key].index.astype(float) * row['seq_len'] * row['bs']
    df.loc[:, 'D'] = Ds
    df.loc[:, 'current_steps'] = current_steps
    df.loc[:, 'portion_steps_elapsed'] = portion_steps_elapsed
    if ckpt:
          df.loc[:,'N'] = df[isoflop_args['n_key']]
          df.loc[:, loss_key] = df[loss_key].values
          df = df.explode([loss_key, 'D', 'current_steps', 'portion_steps_elapsed'])
          df.loc[:, 'C'] = df["D"] * df[flops_field_name]
          df.loc[:, 'loss'] = df[loss_key]
          df = df.astype({'D': 'float64', 'C': 'float64', 'loss': 'float64', 'N': int})

    else:
        for ind, row in df.iterrows():
            df.loc[ind, f'last_{loss_key}_C'] = row[loss_key].index[-1] * row[flops_field_name]
            df.loc[ind, f'last_{loss_key}_D'] = row[loss_key].index[-1]
            df.loc[ind, f'last_{loss_key}'] = row[loss_key].iloc[-1]
            df.loc[ind, 'current_steps'] = row['current_steps'][-1]
            df.loc[ind, 'portion_steps_elapsed'] = row['portion_steps_elapsed'][-1]
        df.loc[:,'N'] = df[isoflop_args['n_key']]
        df.loc[:,'C'] = df[f'last_{loss_key}_C']
        df.loc[:,'D'] = df[f'last_{loss_key}_D']
        df.loc[:,'loss'] = df[f'last_{loss_key}']
        df = df.astype({'D': 'float64', 'C': 'float64', 'loss': 'float64', 'N': int})

    df['N_no_emb'] = precise_param_count_open_lm(df.width, df.depth, vocab_size=0)
    df['C_no_emb'] = df['N_no_emb'] * 6 * df['D']
    df['C_6ND'] = 6 * df['N'] * df['D']
    return df


# epoch_df = pd.read_csv("/Users/margaretli/gitfiles/scaling-app-copy/example/porian.csv")
# epoch_df['D'] = epoch_df['C'] // (  epoch_df['N'] * 6  )
# epoch_df.to_csv("/Users/margaretli/gitfiles/scaling-app-copy/example/porian_edited.csv", index=False)

rsld_df = get_rsld_data('rsld', ckpt=True)
rsld_df.to_csv(f"{DATA_FOLDER}/rsld_with_ckpt_edited.csv", index=False)
rsld_df_no_ckpt = get_rsld_data('rsld', ckpt=False)
rsld_df_no_ckpt.to_csv(f"{DATA_FOLDER}/rsld_without_ckpt_edited.csv", index=False)