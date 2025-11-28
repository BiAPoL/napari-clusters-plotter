import os
import zipfile
from pathlib import Path

def create_sample_data_zip():
    sample_data_dir = Path(__file__).parent
    zip_path = sample_data_dir / "sample_data.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(sample_data_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(sample_data_dir)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    create_sample_data_zip()