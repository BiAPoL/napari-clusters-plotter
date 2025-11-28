import os
import zipfile
from pathlib import Path

def create_sample_data_zip():
    import shutil
    sample_data_dir = Path(__file__).parent / "sample_data"
    zip_path = Path(__file__).parent / "sample_data.zip"

    if os.path.exists(zip_path):
        shutil.rmtree(zip_path)

    if os.path.exists(sample_data_dir.parent / "data_registry.txt"):
        os.remove(sample_data_dir.parent / "data_registry.txt")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(sample_data_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(sample_data_dir)
                zipf.write(file_path, arcname)

    # copy to sample_data folder
    dest_path = sample_data_dir.parent / "sample_data" / "sample_data.zip"
    os.replace(zip_path, dest_path)

def create_registry_file():
    import hashlib
    root_dir = Path(__file__).parent / "sample_data"  # Update this path
    registry = {}
    with open(root_dir / "data_registry.txt", "w") as registry_file:
        for root, _, files in os.walk(root_dir):
            for fn in files:
                fp = Path(root) / fn
                rel_path = fp.relative_to(root_dir).as_posix()
                with open(fp, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                registry[str(rel_path)] = f"sha256:{file_hash}"
                registry_file.write(f'{rel_path}: sha256:{file_hash}\n')

if __name__ == "__main__":
    create_registry_file()
    create_sample_data_zip()
    create_registry_file()