from pathlib import Path
import hashlib
from huggingface_hub import HfApi, CommitOperationAdd, ModelCard

repo = 'iaac-maai/day4_team5_church_style_classifier'
api = HfApi()
info = api.model_info(repo, files_metadata=True)
files = {f.rfilename: f for f in info.siblings}
changed = []
track = Path(__file__).resolve().parents[1]
ModelCard.load(str(track / 'models/README.md'))
names = ['model.pt', 'README.md', 'predict.py', 'classes.json', 'requirements.txt', 'run_metadata.json',
         'experiments.csv', 'training_history.csv', 'test_metrics.csv', 'per_class_test.csv',
         'new_church_predictions.csv', 'figure_credits.csv', 'class_examples.png', 'confusion_matrix.png',
         'misclassified.png', 'new_churches.png', 'test_predictions.png', 'test_vs_baseline.png', 'training_curves.png']
artifacts = [(name, track / 'models' / name) for name in names]
artifacts += [('neural_network.ipynb', track / 'notebook/neural_network.ipynb'),
              ('LICENSE', track.parent / 'LICENSE')]
for name, path in artifacts:
    content = path.read_bytes()
    remote = files.get(name)
    if remote is None:
        same = False
    elif remote.lfs is not None:
        same = remote.lfs.sha256 == hashlib.sha256(content).hexdigest()
    else:
        blob = b'blob ' + str(len(content)).encode() + b'\0' + content
        same = remote.blob_id == hashlib.sha1(blob).hexdigest()
    print(name, 'matches' if same else 'upload needed', flush=True)
    if not same:
        changed.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
result = api.create_commit(repo_id=repo, operations=changed, parent_commit=info.sha,
                           commit_message='Update Track B layout in model card and portable notebook') if changed else None
print('Published revision', result.oid if result else info.sha, flush=True)
print('Repository https://huggingface.co/' + repo, flush=True)
verified = api.model_info(repo, revision=result.oid if result else info.sha, files_metadata=True)
published = {f.rfilename: f for f in verified.siblings}
for name, path in artifacts:
    data = path.read_bytes()
    remote = published[name]
    actual = hashlib.sha256(data).hexdigest() if remote.lfs else hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == (remote.lfs.sha256 if remote.lfs else remote.blob_id), name
print(f'All {len(artifacts)} published artifacts match local files.', flush=True)
