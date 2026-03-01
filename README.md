# Progressive Semantic-Driven Hashing (PSDZH)

Official PyTorch implementation of  
**Progressive Semantic-Driven Hashing for Zero-Shot Image Retrieval**

## Experimental Benchmarks

We evaluate PSDZH on:
- **AWA2**
- **CUB**
- **SUN**
Hash code lengths:
- 16 / 32 / 64 / 128 bits
- Dataset Preparation

Download datasets from official sources and organize them as:

```text
data/
  ├── AWA2/
  ├── CUB/
  └── SUN/
```
Training

Example:
```bash
python train.py --dataset AWA2 --bit 64
```
