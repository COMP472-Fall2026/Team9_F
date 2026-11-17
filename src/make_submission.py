import sys
import pandas as pd

predictions = pd.concat([pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])])
submission = pd.read_csv("data/sample_submission.csv")[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
assert len(submission) == 10000 and submission["label"].notna().all()
assert submission.loc[submission["id"].str.startswith("img_"), "label"].between(0, 49).all()
assert submission.loc[submission["id"].str.startswith("text_"), "label"].between(0, 2).all()
submission.to_csv("submissions/submission.csv", index=False)