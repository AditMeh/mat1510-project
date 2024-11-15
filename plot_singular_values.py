import torch
import matplotlib.pyplot as plt
import tqdm

import numpy as np
from scipy import stats

def marchenko_pastur_pdf(x, y, sigma=1):
    # Lambda plus/minus are the edges of the distribution
    lambda_plus = sigma**2 * (1 + np.sqrt(1/y))**2
    lambda_minus = sigma**2 * (1 - np.sqrt(1/y))**2
    
    # Return zeros for x outside [lambda_minus, lambda_plus]
    pdf = np.zeros_like(x)
    
    # Compute the density for values within the support
    idx = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[idx] = np.sqrt((lambda_plus - x[idx]) * (x[idx] - lambda_minus)) / (2 * np.pi * sigma**2 * y * x[idx])
    
    return pdf


def marchenko_pastur_pdf(x, n, m):
    
    sigma = np.sqrt(n) / np.sqrt(m)
    
    lambda_plus = sigma * (1 + np.sqrt(m/n))
    lambda_minus = sigma * (1 - np.sqrt(m/n))
    
    # Return zeros for x outside [lambda_minus, lambda_plus]
    pdf = np.zeros_like(x)
    
    # Compute the density for values within the support
    idx = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[idx] = np.sqrt((lambda_plus**2 - x[idx]**2) * (x[idx]**2 - lambda_minus**2)) 
    pdf[idx] = pdf[idx] * ((n / m) / (np.pi * sigma**2 * x[idx]))
    
    return pdf

def plot_marchenko_pastur(ax, n, m, num_realizations=1000):
    # Calculate the ratio y = n/m    
    x = np.linspace(0, 5, 1000)
    mp_pdf = marchenko_pastur_pdf(x, n, m)
    ax.plot(x, mp_pdf, 'r-', lw=1, label='Theoretical')


# Method 1: Basic histogram
def plot_histogram(tensor, bins=200, title='Histogram'):
    # Convert tensor to numpy array
    f, ax = plt.subplots(4, 7, figsize=(12, 10), dpi=800)
    f.suptitle(f'{title}: MP distribution vs singular value distribution', fontsize=16)
    for i in tqdm.tqdm(range(4)):
        for j in tqdm.tqdm(list(range(7))):
            idx = i * 7 + j
            data = tensor[idx][1].numpy()
            ax[i, j].hist(data, bins=bins, density=True)
            ax[i, j].set_xlim([0,5])
            ax[i, j].set_ylim([0,2])
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])
            ax[i, j].set_title(f"Layer {idx + 1}")
            plot_marchenko_pastur(ax[i, j], *tensor[idx][0].shape)
    f.savefig(f"{title}.png")
    
    
queries = torch.load("llama_queries.pt")
keys = torch.load("llama_keys.pt")
values = torch.load("llama_values.pt")
output = torch.load("llama_o.pt")


plot_histogram(queries, bins=50, title='Query')
plot_histogram(keys, bins=50, title='Keys')
plot_histogram(values, bins=50, title='Values')
plot_histogram(output, bins=50, title='Output')

