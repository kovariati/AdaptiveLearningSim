# GitHub upload without Git

This repository tree is prepared so that every individual file is below the GitHub browser-upload limit.

1. Create an empty GitHub repository.
2. Extract the supplied GitHub package locally.
3. Upload the extracted repository contents with GitHub's **Add file â†’ Upload files** interface.
4. Preserve the directory structure shown in `README.md`.
5. Commit the uploaded files.
6. After upload, open the **Actions** tab to confirm that the reproducibility test workflow completes successfully.

The large derived EdNet reference bundle is intentionally stored as browser-safe `.part*` files, each below the GitHub browser-upload limit. Users can reconstruct it after cloning or downloading the repository with:

```bash
python raw_pipeline/reference_bundle_parts/assemble_reference_bundle.py
```

The reconstruction script checks the SHA-256 digest automatically.

