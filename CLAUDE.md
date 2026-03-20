# Claude Code Settings

## Permissions

Allow all non-destructive operations without asking, including:
- Reading, writing, and editing files
- Running shell commands (python, bash, docker, tmux, git status/diff/log/add/commit, npm, pip, etc.)
- Creating files and directories
- Running tests and builds
- Git operations: add, commit, checkout, branch, merge, pull, fetch, stash, rebase

Always ask before:
- `git push`, `git push --force`, or any remote-affecting git commands
- Removing files or directories (`rm`, `rm -rf`, `rmdir`, `git clean`)
- Any force flags (`--force`, `-f` on destructive commands, `--hard`)
- `git reset --hard`
- Dropping databases or deleting data
- Killing processes
- Modifying CI/CD pipelines
- Sending messages to external services (Slack, email, GitHub comments/PRs/issues)
