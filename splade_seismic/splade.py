"""Минимальная модель SPLADE (MLM-голова -> log(1+ReLU) -> max-pooling) и кодирование в разреженный формат."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class Splade(nn.Module):
    def __init__(self, name_or_path: str):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained(name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits  # B x L x V
        w = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1)
        return w.max(dim=1).values  # B x V


def tokenizer_for(name_or_path: str):
    return AutoTokenizer.from_pretrained(name_or_path)


@torch.no_grad()
def encode_to_csr(model: Splade, tok, texts, max_len: int, batch_size: int, device: str,
                  min_weight: float = 0.0, fp16: bool = False, log_every: int = 0) -> sp.csr_matrix:
    """Кодирует тексты в scipy CSR (N x V); значения не больше min_weight отбрасываются.

    Тексты кодируются в порядке возрастания длины (меньше паддинга, быстрее), строки результата
    возвращаются в исходном порядке. fp16=True включает autocast на GPU (примерно вдвое быстрее).
    """
    model.eval()
    order = np.argsort([len(x) for x in texts], kind="stable")
    rows = []
    use_amp = fp16 and str(device).startswith("cuda")
    for n, i in enumerate(range(0, len(texts), batch_size)):
        chunk = [texts[j] for j in order[i:i + batch_size]]
        batch = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            w = model(batch["input_ids"], batch["attention_mask"])
        w = w.float().cpu().numpy()
        w[w <= min_weight] = 0.0
        rows.append(sp.csr_matrix(w))
        if log_every and (n + 1) % log_every == 0:
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    mat = sp.vstack(rows).tocsr()
    return mat[np.argsort(order)]  # обратно в исходный порядок
