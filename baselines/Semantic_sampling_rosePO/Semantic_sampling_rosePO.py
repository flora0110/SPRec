import json
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
def process_batch(batch):
    results = []
    for data in batch:
        input = data['input']
        names = re.findall(r'"([^"]+)"', input)
        name_embeddings = torch.tensor([model.encode(name) for name in names], device="cuda")
        cosine_similarity = F.cosine_similarity(name_embeddings[:, None, :], embeddings[None, :, :], dim=-1)
        similarity = cosine_similarity.mean(dim=0)
        min_sim, min_index = similarity.min(dim=-1)
        semantic_item = id2name[str(min_index.item())]
        data['semantic'] = f"\"{semantic_item}\"\n"
        results.append(data)
    return results
model = SentenceTransformer('./models/paraphrase-MiniLM-L3-v2')
def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)
def export_to_json(file_path:str,dic):
    f = open(file_path, 'w')
    json.dump(dic,f,indent=2)
# semantic item
for category in ["CDs_and_Vinyl"]:
    embeddings = torch.load(f"../eval/{category}/embeddings.pt").to('cuda')
    id2name = read_json(f"../eval/{category}/id2name.json")
    train_data = read_json(f"./{category}/train.json")
    batch_size = 64
    batched_data = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
    final_data = []
    for batch in tqdm(batched_data, desc=f"Processing {category} train data......"):
        final_data.extend(process_batch(batch))
    export_to_json(f"../data/{category}/train_semantic.json",train_data)