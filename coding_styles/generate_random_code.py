from datasets import load_dataset

ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["javascript"], trust_remote_code=True)
print(next(iter(ds))["code"])
