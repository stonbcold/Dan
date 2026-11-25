import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from common import SOURCES, raw_dir

BYTES_PER_TOKEN = 3.5
MARGIN = 1.2


def text_bytes(path):
    meta = pq.ParquetFile(path).metadata
    col = meta.schema.names.index("text")
    return sum(meta.row_group(i).column(col).total_uncompressed_size
               for i in range(meta.num_row_groups))


def download(name):
    api = HfApi()
    for repo, prefix, target in SOURCES[name]:
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                       if f.startswith(prefix) and f.endswith(".parquet"))
        need = target * BYTES_PER_TOKEN * MARGIN
        got = 0
        for k, f in enumerate(files):
            path = hf_hub_download(repo, f, repo_type="dataset", local_dir=raw_dir(name))
            got += text_bytes(path)
            print(f"[{name}] {f}  ->  {got / 1e9:.1f} / {need / 1e9:.1f} Go de texte", flush=True)
            if got >= need:
                break
        else:
            print(f"[{name}] ATTENTION : {prefix} epuise avant l'objectif ({got / 1e9:.1f} Go)")


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        download(name)
