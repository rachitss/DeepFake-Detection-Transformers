import os
import json
import shutil
import pandas as pd


TRAIN_DIR = 'D:\\W\\VS\\VS Folder\\DFD\\DFDC\\deepfake-detection-challenge\\train_sample_videos\\'
METADATA_PATH = os.path.join(TRAIN_DIR, 'metadata.json')
OUTPUT_DIR = 'D:\\W\\VS\\VS Folder\\DFD\\DFDC\\deepfake-detection-challenge\\real-fake-split\\'


def build_metadata_frame(metadata_dict):
    return pd.DataFrame(
        [
            (
                video_file,
                video_info['label'],
                video_info.get('split', ''),
                video_info.get('original', ''),
            )
            for video_file, video_info in metadata_dict.items()
        ],
        columns=['filename', 'label', 'split', 'original'],
    )


def ensure_directories():
    os.makedirs(os.path.join(OUTPUT_DIR, 'real'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'fake'), exist_ok=True)


def copy_videos(metadata_dict):
    for filename, video_info in metadata_dict.items():
        src = os.path.join(TRAIN_DIR, filename)
        label = video_info.get('label', '').lower()
        if label not in {'real', 'fake'}:
            continue

        dest_dir = os.path.join(OUTPUT_DIR, label)
        dest = os.path.join(dest_dir, filename)

        if not os.path.exists(src):
            print(f'Source missing: {src}')
            continue
        if os.path.exists(dest):
            continue

        shutil.copy2(src, dest)


def main():
    with open(METADATA_PATH, 'r') as metadata_file:
        metadata_dict = json.load(metadata_file)

    train_df = build_metadata_frame(metadata_dict)
    print(train_df.head())

    ensure_directories()
    copy_videos(metadata_dict)
    print('Real/fake split populated at', OUTPUT_DIR)


if __name__ == '__main__':
    main()