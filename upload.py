from huggingface_hub import HfApi
import os
api = HfApi(token="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")
api.upload_folder(
    folder_path="/media/raid/workspace/zhaoyanpeng/model/maze_dataset",
    repo_id="piekenius123/Amaze",
    repo_type="dataset",
)