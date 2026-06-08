# Project Codex Rules

## Path Reporting Absolute Rule

- When the user asks for a file, folder, model, result, output, log, image, dataset, or any other path, answer with the full absolute filesystem path.
- Do not answer with project-relative paths such as `runs/...`, `data/...`, or `result_grouping/...` unless they are accompanied by the full absolute path.
- For this project, prefer paths rooted at `D:\project\unknown-contrastive\...` when the artifact is inside the repository.
- If multiple paths are relevant, list each one as a full absolute path.
