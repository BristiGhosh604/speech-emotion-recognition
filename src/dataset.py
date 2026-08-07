import os

# Maps the RAVDESS filename emotion code to a readable label
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_emotion_from_filename(filename):
    """
    RAVDESS filenames look like: 03-01-02-01-01-01-01.wav
    The 3rd number (index 2 after splitting) is the emotion code.
    """
    parts = filename.replace(".wav", "").split("-")
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, "unknown")


def load_dataset(raw_dir):
    """
    Walks through data/raw/Actor_XX/ folders and returns a list of
    (file_path, emotion_label) pairs for every .wav file found.
    """
    samples = []
    for actor_folder in os.listdir(raw_dir):
        actor_path = os.path.join(raw_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                label = parse_emotion_from_filename(filename)
                full_path = os.path.join(actor_path, filename)
                samples.append((full_path, label))
    return samples


if __name__ == "__main__":
    samples = load_dataset("data/raw")
    print(f"Total samples found: {len(samples)}")
    print("First 5 samples:")
    for path, label in samples[:5]:
        print(f"  {label:10s} -> {path}")