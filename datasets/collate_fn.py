import torch
import torch.nn.functional as F


def collate_fn_same_length(batch):
    all_queries = []
    docs = []
    for queries, doc in batch:
        all_queries.extend(queries)
        docs.append(doc)
    queries_tensor = torch.stack(all_queries, dim=0)
    docs_tensor = torch.stack(docs, dim=0)
    return queries_tensor, docs_tensor


def collate_fn_dif_length(batch):
    all_queries = []
    docs = []

    for queries, doc_spec in batch:
        queries = [q.unsqueeze(0) if q.dim() == 2 else q for q in queries]
        doc_spec = doc_spec.unsqueeze(0) if doc_spec.dim() == 2 else doc_spec
        all_queries.extend(queries)
        docs.append(doc_spec)

    query_times = [q.shape[-1] for q in all_queries]
    max_query_time = max(query_times)
    padded_queries = []
    for q in all_queries:
        pad_amount = max_query_time - q.shape[-1]
        padded_q = F.pad(q, (0, pad_amount))
        padded_queries.append(padded_q)
    queries_tensor = torch.stack(padded_queries, dim=0)

    doc_times = [d.shape[-1] for d in docs]
    max_doc_time = max(doc_times)
    padded_docs = []
    for d in docs:
        pad_amount = max_doc_time - d.shape[-1]
        padded_d = F.pad(d, (0, pad_amount))
        padded_docs.append(padded_d)
    docs_tensor = torch.stack(padded_docs, dim=0)

    return queries_tensor, docs_tensor
