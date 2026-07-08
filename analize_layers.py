import os                                                                                                                                                                                                
from collections import defaultdict                                                                                                                                                               
from safetensors import safe_open                                                                                                                                
                                                                                                                                                                   
d = '/home/stoyko/Storage/VLLM/nvidia/Qwen3.6-35B-A3B-NVFP4-Source'                                                                                              
                                                                                                                                                                                                           
DTYPE_BYTES = {                                                                                                                                                  
    'torch.float32': 4, 'torch.float16': 2, 'torch.bfloat16': 2,                                                                                                 
    'torch.float8_e4m3fn': 1, 'torch.float8_e5m2': 1, 'torch.uint8': 1,                                                                                                                                  
    'torch.int8': 1, 'torch.int16': 2, 'torch.int32': 4, 'torch.int64': 8,                                                                                                                               
}                                                                                                                                                                                                        
                                                                                                                                                                                                           
def fmt(n):                                                                                                                                                                                              
    for u in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):                                                                                                                                                          
        if abs(n) < 1024:                                                                                                                                                                                
            return f"{n:>7.2f} {u}" if u != 'B' else f"{n:>7} B"                                                                                                                                         
        n /= 1024                                                                                                                                                                                        
    return f"{n:.2f} PiB"                                                                                                                                                                                
                                                                                                                                                                                                           
def dtype_str(t):
    return str(t.dtype).split('.')[-1]                                                                                                                                                                   
                                                                                                                                                                                                         
# Pattern-based category classifier                                                                                                                                                                      
def classify(name):                                                                                                                                                                                      
    if 'embed_tokens' in name:                                                                                                                                                                           
        return 'Embedding'                                                                                                                                                                               
    if 'lm_head' in name and ('weight_scale' in name or 'input_scale' in name or 'output_scale' in name):                                                                                                
        return 'Quant Metadata'                                                                                                                                                                          
    if 'lm_head' in name:                                                                                                                                                                                
        return 'LM Head'                                                                                                                                                                                 
    if 'shared_expert' in name:                                                                                                                                                                          
        return 'Shared Expert'                                                                                                                                                                           
    if 'experts.' in name and ('down_proj' in name or 'gate_up_proj' in name or 'up_proj' in name):                                                                                                      
        return 'MoE Expert Core'                                                                                                                                                                         
    if 'experts.' in name:                                                                                                                                                                               
        return 'MoE Expert Metadata'                                                                                                                                                                     
    if 'router' in name.lower():                                                                                                                                                                         
        return 'Router'                                                                                                                                                                                  
    if '.mlp.' in name or 'mlp.' in name:                                                                                                                                                                
        if 'input_scale' in name or 'weight_scale' in name or 'weight_scale_2' in name:                                                                                                                  
            return 'Quant Metadata'
        return 'MLP'                                                                                                                                                                                     
    if 'self_attn' in name:                                                                                                                                                                              
        if 'input_scale' in name or 'weight_scale' in name:                                                                                                                                              
            return 'Quant Metadata'                   
        if 'linear_attn' in name:                                                                                                                                                                        
            return 'Linear Attention'                                                                                                                           
        return 'Attention'                                                                                                                                      
    if 'input_layernorm' in name or 'post_attention_layernorm' in name or 'norm' in name:                                                                       
        return 'Layer Norm'                                                                                                                                     
    if name.startswith('mtp'):                                                                                                                                                                           
        if 'input_scale' in name or 'weight_scale' in name:                                                                                                     
            return 'Quant Metadata'                                                                                                                             
        return 'MTP'                                                                                                                                                                                     
    if name.startswith('model.visual') or name.startswith('vision'):                                                                                                                                     
        return 'Vision'                                                                                                                                                                                  
    return 'Other'                                                                                                                                                                                       
                                                                                                                                                                                                           
records = []                                                                                                                                                                                             
total_bytes = 0                                                                                                                                                                                          
                                                                                                                                                                                                         
for fname in sorted(f for f in os.listdir(d) if f.endswith('.safetensors')):                                                                                                                             
    with safe_open(os.path.join(d, fname), framework='pt') as f:                                                                                                                                         
        for k in f.keys():                                                                                                                                                                               
            t = f.get_tensor(k)                                                                                                                                                                          
            dt = dtype_str(t)                                                                                                                                                                            
            nbytes = t.numel() * DTYPE_BYTES.get(dt, 4)                                                                                                                                                  
            total_bytes += nbytes                                                                                                                                                                        
            records.append((k, list(t.shape), dt, nbytes, classify(k)))                                                                                                                                  
                                                                                                                                                                                                         
records.sort(key=lambda x: -x[3])                                                                                                                                                                        
                                                                                                                                                                                                         
# ── Top 30 tensors ──                                                                                                                                                                                   
print("Top 30 tensors by size:")                                                                                                                                                                         
print(f"{'Tensor':<72} {'Shape':<28} {'Dtype':<14} {'Bytes':>12} {'%':>6}  {'Category':<20}")                                                                                                            
print('-' * 158)                                                                                                                                                                                         
for name, shape, dt, nbytes, cat in records[:30]:                                                                                                                                                        
    pct = 100.0 * nbytes / total_bytes                                                                                                                                                                   
    s = str(shape)                                                                                                                                                                                       
    print(f"{name:<72} {s:<28} {dt:<14} {fmt(nbytes):>12} {pct:>5.2f}%  {cat:<20}")                                                                                                                      
                                                                                                                                                                                                         
print(f"\n{'TOTAL':<72} {'':<28} {'':<14} {fmt(total_bytes):>12} {'100.00%':>6}")                                                                                                                        
print(f"Checkpoint total: {fmt(total_bytes)} ({total_bytes/1024**3:.2f} GiB on disk w/ overhead)")     
                                                                                                                                                                                                         
# ── Categories ──                                                                                                                                                                                       
cat_bytes = defaultdict(int)                                                                                                                                                                             
cat_tensors = defaultdict(int)                                                                                                                                                                           
for _, _, _, nbytes, cat in records:                                                                                                                                                                     
    cat_bytes[cat] += nbytes                                                                                                                                                                             
    cat_tensors[cat] += 1                                                                                                                                                                                
                                                                                                                                                                                                         
print(f"\n{'Category':<30} {'Tensors':>8} {'Bytes':>14} {'%':>8}")                                                                                                                                       
print('-' * 62)                                                                                                                                                                                          
sorted_cats = sorted(cat_bytes.items(), key=lambda x: -x[1])                                                                                                                                             
for cat, nbytes in sorted_cats:                                                                                                                                                                          
    pct = 100.0 * nbytes / total_bytes                                                                                                                                                                   
    print(f"{cat:<30} {cat_tensors[cat]:>8} {fmt(nbytes):>14} {pct:>7.2f}%")                                                                                                                             
print('-' * 62)                                                                                                                                                                                          
print(f"{'TOTAL':<30} {len(records):>8} {fmt(total_bytes):>14} {'100.00%':>8}")                                                                                                                     
                                                       
