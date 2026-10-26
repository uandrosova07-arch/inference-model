from pathlib import Path
import subprocess
import tarfile

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "msmarco"
BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
FILES = ["qrels.dev.tsv", "queries.tar.gz", "collection.tar.gz"]

def download(name):
    path = DATA_DIR / name
    print(f"Downloading {name}...", flush=True)
    subprocess.run([
        "curl", "-fL", "--retry", "10",
        "--retry-delay", "5", "-C", "-",
        "-o", str(path), f"{BASE_URL}/{name}"
    ], check=True)
    return path

def extract(archive):
    print(f"Extracting {archive.name}...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter="data")
    archive.unlink()

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if path.exists():
            print(f"Skipping {name}", flush=True)
        else:
            download(name)
        if name.endswith(".tar.gz"):
            extract(path)
    print("MS MARCO ready!")

if __name__ == "__main__":
    main()
