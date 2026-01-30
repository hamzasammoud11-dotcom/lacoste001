import os, pickle, numpy as np, torch
from rdkit import Chem
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from DeepPurpose import DTI as dp_models
from DeepPurpose.utils import smiles2morgan
from config import BEST_MODEL_RUN, QDRANT_HOST, QDRANT_PORT

SRC = "molecules_v2"
DST = "molecules_dp"
BATCH = 64

cfg_path = os.path.join(BEST_MODEL_RUN, "config.pkl")
model_path = os.path.join(BEST_MODEL_RUN, "model.pt")

cfg = pickle.load(open(cfg_path, "rb"))
cfg["result_folder"] = BEST_MODEL_RUN

m = dp_models.model_initialize(**cfg)
m.load_pretrained(model_path)

import DeepPurpose.encoders as dp_encoders
device = torch.device("cpu")
dp_encoders.device = device
m.model = m.model.to(device)
m.model.eval()

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

offset = None
total = 0

while True:
    pts, offset = client.scroll(
        collection_name=SRC,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    if not pts:
        break

    batch = []
    for p in pts:
        payload = p.payload or {}
        smiles = payload.get("smiles") or payload.get("content")
        if not smiles:
            continue
        if Chem.MolFromSmiles(smiles) is None:
            continue

        fp = smiles2morgan(smiles, radius=2, nBits=1024)
        if fp is None:
            continue

        v = torch.tensor(np.array([fp]), dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = m.model.model_drug(v).detach().cpu().numpy()[0].astype(float).tolist()

        batch.append(PointStruct(id=p.id, vector={"molecule": emb}, payload=payload))

        if len(batch) >= BATCH:
            client.upsert(collection_name=DST, points=batch, wait=True)
            total += len(batch)
            print("upserted", total)
            batch = []

    if batch:
        client.upsert(collection_name=DST, points=batch, wait=True)
        total += len(batch)
        print("upserted", total)

    if offset is None:
        break

print("done", total)
