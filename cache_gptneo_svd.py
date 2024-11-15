
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm

# Specify the path to your local model
# model_path = "/model-weights/Llama-3.2-3B"
model_path = "rhubarbwu/TinyStories-12x1024_10L"

# Load the tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="float32",
).to(torch.float32)

def get_query_svds(module_list):
    svds_queries = []
    svds_keys = []
    svds_values = []
    svds_o = []
    
    for layer in tqdm.tqdm(module_list):
        q_weight = layer.attn.attention.q_proj.weight
        U, S, V = torch.linalg.svd(q_weight.detach())
        svds_queries.append((U, S, V))

        k_weight = layer.attn.attention.k_proj.weight
        U, S, V = torch.linalg.svd(k_weight.detach())
        svds_keys.append((U, S, V))

        v_weight = layer.attn.attention.v_proj.weight
        U, S, V = torch.linalg.svd(v_weight.detach())
        svds_values.append((U, S, V))
        
        o_weight = layer.attn.attention.out_proj.weight
        U, S, V = torch.linalg.svd(o_weight.detach())
        svds_o.append((U, S, V))

    return svds_queries, svds_keys, svds_values, svds_o

svds_queries, svds_keys, svds_values, svds_o = get_query_svds(model.transformer.h)


torch.save(svds_queries, "tinystories_queries.pt")
torch.save(svds_keys, "tinystories_keys.pt")
torch.save(svds_values, "tinystories_values.pt")
torch.save(svds_o, "tinystories_o.pt")
