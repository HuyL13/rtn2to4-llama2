"""Load existing IF-SFT Arrow rows despite newer HF List schema metadata."""
import json
from pathlib import Path


def load_fingerprint_dataset(path):
    from datasets import Dataset, DatasetDict, load_from_disk
    import pyarrow as pa

    path = Path(path)
    try:
        return load_from_disk(str(path))
    except TypeError as error:
        if 'must be called with a dataclass type or instance' not in str(error):
            raise
        result = {}
        splits = json.loads((path / 'dataset_dict.json').read_text())['splits']
        for split in splits:
            split_path = path / split
            state = json.loads((split_path / 'state.json').read_text())
            if state.get('_indices_data_files'):
                raise RuntimeError('Indexed datasets require explicit migration') from error
            rows = []
            for entry in state['_data_files']:
                with pa.memory_map(str(split_path / entry['filename']), 'r') as source:
                    # Read the actual rows, avoiding the incompatible HF schema metadata.
                    rows.extend(pa.ipc.open_stream(source).read_all().to_pylist())
            result[split] = Dataset.from_list(rows)
        print('Loaded IF-SFT Arrow rows with compatible feature metadata', flush=True)
        return DatasetDict(result)
