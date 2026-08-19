# The `todo/` Folder

A drop box for handing files to coding agents. Copy something in (an export from
another tool, a download, a doc someone sent you), point the agent at it, and the agent
reads it, acts on it, and files or deletes it. The resting state of this folder is empty.

Contents are gitignored. The folder and this README are tracked, so the convention
survives a fresh clone.

## How It Works

1. Drop the file in `todo/`.
2. Reference it in a prompt: `@todo/whatever.md`.
3. The agent reads it and does the work.
4. The agent moves it to its real home in the repo, or deletes it. Nothing stays.

## Rules

- **The repo is the destination, not this folder.** Whatever is worth keeping gets moved
  into a tracked location. Treat everything sitting here as in transit.
- **Never delete the folder or this README.** This file is the only tracked thing in
  here. Removing it drops the folder from fresh clones and takes the convention with it.
- **Agents: this is not your working space.** What an agent generates on its own goes
  wherever its session scratchpad lives. This folder holds what a human put in it.
- **Gitignore keeps a file out of commits, not off disk.** If you drop something with
  credentials in it, clearing the folder is the only thing that removes it.

## Setup

> [!IMPORTANT]
> The pattern must be `todo/*`, not `todo/`. Git cannot re-include a file whose parent
> directory is excluded, so `todo/` paired with `!todo/README.md` silently drops the
> README and the folder disappears from clones.

```gitignore
# Drop box for files handed to agents; keep the folder and its README
todo/*
!todo/README.md
```

## Where This Comes From

Bootstrap seeds this folder. A run writes this README when `todo/README.md` is
absent and the upstream template is available, creating the folder if it is not
there, and adds the two `.gitignore` lines above when they are missing. A repo
therefore picks up the convention on its first session and a new clone picks it
up on the next one. `AGENT_CONFIG_NO_TODO_DROPBOX=1` turns the whole step off.

An existing README is never rewritten. A repo whose filing rules are specific to
its own work should say so here, and bootstrap leaves that edit alone. Deleting
the file is what brings the upstream copy back.
