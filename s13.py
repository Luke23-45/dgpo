import lmdb

db_path = r"data/demos/expert_expert_run_v1_20k_samples_2939e707515a1a6963254bc55001a1b0.lmdb"

env = lmdb.open(
    db_path,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    subdir=False  # 👈 CRUCIAL FIX on Windows!
)

with env.begin() as txn:
    print("✅ Successfully opened LMDB")
    print("Entries:", txn.stat()['entries'])

env.close()
