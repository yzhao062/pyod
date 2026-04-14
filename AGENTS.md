<!-- Quick start: In Claude Code, read @AGENTS.md to run bootstrap, session checks, and task routing -->

> **Claude Code / Codex / other agents -- determine your context before proceeding:**
>
> **Source repo test:** If `bootstrap/bootstrap.sh`, `bootstrap/bootstrap.ps1`, and `reference-skills/` all exist at the repo root, you are in the `agent-config` source repo (or a fork with the same layout). **Skip the bootstrap block below** and proceed directly to `## Session Start Check`. Do not execute the bootstrap script in the source repo; it would try to bootstrap itself.
>
> **Consumer repo path:** Otherwise, you are in a project that consumes the shared config. **You MUST execute the matching bootstrap block below at the start of every session, before responding to any user request or running session checks.** Use the PowerShell block on Windows and the Bash block on macOS/Linux. Bootstrap is idempotent: run it every session even if `.agent-config/` already exists, so shared config, skills, and settings are refreshed from upstream before any work begins. Do not treat the block below as reference documentation -- it is a required runtime command. After running, report the refresh result in one line (for example, "bootstrap: refreshed AGENTS.md, skills, settings").

## Bootstrap: Shared Config and Skills

This file is the shared agent configuration hosted at `https://github.com/yzhao062/agent-config`.
Project repos should include the following bootstrap block in their own `AGENTS.md` so that agents automatically pull the latest shared defaults and skills before starting any task. Because bootstrap overwrites the consumer's root `AGENTS.md` with this source on every run, the block below ends up serving two roles in consumer repos: it is both the embedded reference for how the setup works and the runtime command the agent executes each session (see the top-of-file note for the source-vs-consumer test that decides which role applies in the current repo).

### Bootstrap block for project repos

Copy the block below into the top of each project repo's `AGENTS.md`. Because bootstrap overwrites `AGENTS.md` on every run, put project-local overrides in `AGENTS.local.md`.

````
## Shared Agent Config (auto-fetched)

Before starting any task, fetch the shared agent configuration and skills. Use the block for your shell.

PowerShell (Windows):

```powershell
New-Item -ItemType Directory -Force -Path .agent-config, .claude, .claude/commands | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri https://raw.githubusercontent.com/yzhao062/agent-config/main/bootstrap/bootstrap.ps1 -OutFile .agent-config/bootstrap.ps1
& .\.agent-config\bootstrap.ps1
```

Bash (macOS/Linux):

```bash
mkdir -p .agent-config .claude/commands
curl -sfL https://raw.githubusercontent.com/yzhao062/agent-config/main/bootstrap/bootstrap.sh -o .agent-config/bootstrap.sh
bash .agent-config/bootstrap.sh
```

This bootstrap flow refreshes the consuming repo's root `AGENTS.md` to match the shared copy. If a project later needs repo-local overrides, put them in `AGENTS.local.md`.

Read and follow the rules in `.agent-config/AGENTS.md` as baseline defaults. Any rule in `AGENTS.local.md` overrides the shared default.
When a skill is invoked, read its SKILL.md from `.agent-config/repo/skills/<skill-name>/SKILL.md`.
If a local `skills/<skill-name>/SKILL.md` exists in the project repo, the local copy takes precedence.
Copying `.agent-config/repo/.claude/commands/*.md` only overwrites command files with the same name as the shared repo and does not delete unrelated project-local commands.
Merge shared Claude project defaults (e.g., `permissions`, `attribution`) from `.agent-config/repo/.claude/settings.json` into the project `.claude/settings.json`. Shared keys are updated on every bootstrap run; project-only keys are preserved. Merge requires Python; if unavailable the existing file is left untouched.
Add `.agent-config/` to the project's `.gitignore` so fetched files are not committed.
Bootstrap also sets up user-level config: it copies `scripts/guard.py` to `~/.claude/hooks/` (a PreToolUse hook that guards against destructive commands) and merges `user/settings.json` into `~/.claude/settings.json` (shared permissions, hook wiring, and the `CLAUDE_CODE_EFFORT_LEVEL=max` env entry that sets the default effort level). Remove the user-level section from the bootstrap script if this is not wanted.
````

### What gets shared

| Content | Source | How fetched |
|---------|--------|-------------|
| User profile, writing defaults, formatting rules, environment notes | `AGENTS.md` (this file) | `curl` raw file |
| Shared skills (`dual-pass-workflow`, `bibref-filler`, `figure-prompt-builder`, etc.) | `skills/` directory (committed only) | sparse `git clone` |
| Claude pointer commands for shared skills | `.claude/commands/` | sparse `git clone` plus non-destructive copy into the project `.claude/commands/` |
| Claude project defaults (`permissions`, `attribution`, etc.) | `.claude/settings.json` | sparse `git clone` plus key-level merge into the project `.claude/settings.json` on every run |

### Override rules

- If `AGENTS.local.md` exists in the project root, read and follow it after `AGENTS.md`. Rules in `AGENTS.local.md` override the shared defaults.
- Rules in `AGENTS.local.md` always win over shared defaults. Do not edit the root `AGENTS.md` for local overrides, as bootstrap will overwrite it.
- Project-local `skills/<name>/SKILL.md` always wins over the shared copy of the same skill.
- Shared keys in `.claude/settings.json` are updated on every bootstrap run. Project-only keys are preserved. To override a shared key locally, use `.claude/settings.local.json`.
- If a shared skill does not exist locally, the agent should use the fetched copy from `.agent-config/repo/skills/`.

---

<!-- Everything above this line is bootstrap setup instructions. -->
<!-- Everything below this line contains the shared rules that agents should read and follow. -->

## Session Start Check

After bootstrap, run **all** of the following checks and report results in a short summary. No shell commands are needed — all information is available from session environment and config files. Only flag items that need attention — if everything is correct, a one-line confirmation is sufficient.

1. **OS** -- Read the platform from the session environment (e.g., `win32`, `darwin`, `linux`). Note it for platform-specific behavior (e.g., terminal review path on Windows, MCP on macOS/Linux).
2. **Claude Code model and effort** (Claude Code sessions only) -- If the live session environment exposes model name and effort level, check them. The user prefers the highest available model (currently Opus) at max effort. If the session is on a different model or effort, mention it once — this is a preference, not a misconfiguration.
3. **Codex config** -- Read `~/.codex/config.toml` (or `%USERPROFILE%\.codex\config.toml` on Windows). If the file exists, check these keys and report any that are missing or wrong:
   - `model` should be `"gpt-5.4"` (or the latest available)
   - `model_reasoning_effort` should be `"xhigh"`
   - `service_tier` should be `"fast"`
   - `[features] fast_mode` should be `true`
   
   If the file does not exist and Codex is expected, note that too.

## User Profile

- These are user-level defaults that can be reused across projects unless a local repo rule or task-specific instruction is stricter.
- The user is a computer scientist and professor working in machine learning and AI.
- Common tasks include research papers, funding proposals, scientific writing, and administrative writing.

## Agent Roles

- **Claude Code** is the primary workhorse: drafting, implementation, research, and heavy-lifting tasks.
- **Codex** is the gatekeeper: review, feedback, and quality checks on work produced by Claude Code or the user.
- When both agents are available, default to this division of labor unless the user overrides it.

## Task Routing

- Before starting a task, read the router skill to determine which domain skill to use. Look for it in this order: `skills/my-router/SKILL.md` (repo-local), then `.agent-config/repo/skills/my-router/SKILL.md` (bootstrapped from shared config).
- The router inspects prompt keywords, file types, and project structure to dispatch automatically. Do not ask the user which skill to use when the routing table provides a clear match.
- If the `superpowers` plugin is active, the router operates during the execution phase. Superpowers handles the outer workflow (brainstorm, plan, execute, verify); the router handles inner dispatch to the right domain skill.
- If routing is ambiguous (multiple skills could apply), state the detected context and proposed skill, then ask the user to confirm.

## Codex MCP Integration

- Codex is available to Claude Code as an MCP server. Register it once at the user level so it applies to all projects and terminals (including PyCharm):
  ```
  claude mcp add codex -s user -- codex mcp-server -c approval_policy=on-failure
  ```
- This writes to `~/.claude.json` top-level `mcpServers`. A session restart is required after registration for `/mcp` to pick it up.
- **Migrating an existing registration:** If Codex was registered without `-c approval_policy=on-failure`, remove and re-add:
  ```
  claude mcp remove codex -s user
  claude mcp add codex -s user -- codex mcp-server -c approval_policy=on-failure
  ```
  On Windows, adjust the path as shown below.
- **Gotcha:** Do not register under a project scope (e.g., from a specific working directory without `-s user`). That creates a project-scoped entry under `projects["<path>"].mcpServers` in `~/.claude.json`, which does not propagate to other directories.
- Prerequisites: Node.js installed, Codex CLI installed (`npm install -g @openai/codex`), and `OPENAI_API_KEY` set.
- **Recommended Codex defaults (as of April 2026):** Add or update these keys in `~/.codex/config.toml` on macOS/Linux or `%USERPROFILE%\.codex\config.toml` on Windows (create the file if it does not exist) so that both interactive sessions and the MCP server use the recommended default model with fast inference:
  ```toml
  model = "gpt-5.4"
  model_reasoning_effort = "xhigh"
  service_tier = "fast"

  [features]
  fast_mode = true
  ```
  `service_tier = "fast"` selects the fast inference tier (1.5x speed, no quality reduction). For ChatGPT-authenticated users this costs 2x credits; API-key users pay standard API pricing. The `[features].fast_mode` flag gates the feature and defaults to `true`; set it explicitly alongside `service_tier` to persist the default in `config.toml`. Omit both if you prefer lower cost over latency. The MCP server reads the same `config.toml`, so these settings apply to both interactive sessions and MCP. These settings work identically on macOS, Linux, and Windows.
- MCP tools available after registration: `codex` (new prompt) and `codex-reply` (continue an existing session).
- **Windows note:** Claude Code launches MCP servers through bash, not cmd or PowerShell. This means `.cmd` wrappers and PowerShell variables like `$env:APPDATA` do not work. If `codex` is not on `PATH`, use the full path with forward slashes and **no `.cmd` extension** (npm installs a bash-compatible script alongside the `.cmd`):
  ```
  claude mcp add codex -s user -- C:/Users/<you>/AppData/Roaming/npm/codex mcp-server -c approval_policy=on-failure
  ```
  Run `where codex` (cmd) or `Get-Command codex` (PowerShell) to find the actual path.
- **MCP approval policy:** By default the Codex MCP server prompts for approval on every shell command, which surfaces as "MCP server requests your input" dialogs in Claude Code. Pass `-c approval_policy=on-failure` in the registration command (shown above) so commands auto-approve and only prompt on failures. The same key can be set in `config.toml` (`approval_policy = "on-failure"`) for interactive sessions.
- **Bitdefender false positives (Windows):** Bitdefender Advanced Threat Defense may flag Codex and Claude Code shell commands as "Malicious command lines detected." To suppress this, add exceptions in Bitdefender → Protection → Manage Exceptions. For each exception, enable the **Advanced Threat Defense** toggle (not just Antivirus). Recommended exceptions:
  - `C:\Program Files\nodejs\node.exe` (process)
  - `C:\Users\<you>\.local\bin\claude.exe` (process)
  - `C:\Users\<you>\AppData\Roaming\npm\codex` (process)
  - `C:\Users\<you>\AppData\Roaming\npm\codex.cmd` (process)
  - `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe` (process, if Codex invokes PowerShell)
- **Windows recommendation: use the terminal path.** On Windows (11 Build 26200+), the MCP path still has rough edges — residual approval prompts and Bitdefender false positives add friction even after the mitigations above. The terminal path (relay reviews via the Codex interactive terminal window) avoids both issues. Prefer the terminal path on Windows; use MCP on macOS/Linux where it works smoothly.

## Writing Defaults

- Use scientifically accessible language.
- Do not oversimplify unless the user asks for simplification.
- Keep meaningful technical detail.
- Keep factual accuracy and clarity high in scientific contexts.
- Use consistent terms. If an abbreviation is defined once, do not define it again later.
- If citing papers, verify that they exist.
- When paper citations are requested, provide BibTeX entries that can be copied into a `.bib` file.
- Provide code only when necessary. Confirm that the code is correct and can run as written.
- For NSF or other federal proposal work, do not introduce DEI-related terms unless the solicitation explicitly requires them.
- For non-federal proposals or calls that explicitly request DEI framing or terminology, follow the call requirements instead of applying a blanket ban.
- Avoid the following words and close variants unless the user explicitly asks for them: `encompass`, `burgeoning`, `pivotal`, `realm`, `keen`, `adept`, `endeavor`, `uphold`, `imperative`, `profound`, `ponder`, `cultivate`, `hone`, `delve`, `embrace`, `pave`, `embark`, `monumental`, `scrutinize`, `vast`, `versatile`, `paramount`, `foster`, `necessitates`, `provenance`, `multifaceted`, `nuance`, `obliterate`, `articulate`, `acquire`, `underpin`, `underscore`, `harmonize`, `garner`, `undermine`, `gauge`, `facet`, `bolster`, `groundbreaking`, `game-changing`, `reimagine`, `turnkey`, `intricate`, `trailblazing`, `unprecedented`.

## Formatting Defaults

- Preserve the original format when the input is in LaTeX, Markdown, or reStructuredText.
- Do not convert paragraphs into bullet points unless the user asks for that format.
- Prefer full forms such as `it is` and `he would` rather than contractions.
- `e.g.,` and `i.e.,` are fine when appropriate.
- Do not use Unicode character `U+202F`.
- Avoid heavy dash use. Do not use em dashes (`—`) or en dashes (`–`) as casual sentence punctuation. Prefer commas, semicolons, colons, or parentheses instead. En dashes in numeric ranges (e.g., `1–3`, `2020–2025`), paired names, or citations are fine. Normal hyphenation in compound words and technical terms (e.g., `command-line`, `co-PI`, `zero-shot`) is fine and should not be avoided.
- Break extremely long or complex sentences into shorter, more readable ones. If a sentence has multiple clauses or nested qualifications, split it.
- Vary sentence length and structure. Prefer not to start several consecutive sentences with the same word or phrase. Avoid overusing transition words like "Additionally" or "Furthermore." Not every paragraph needs a tidy summary sentence at the end. Mix short, direct sentences with longer ones to keep the writing natural.

## Git Safety

- **Never run `git commit` or `git push` without explicit user approval.** Always show the proposed action and ask for confirmation before executing.
- This rule is non-negotiable and applies to all projects that consume this shared config.
- This includes any variant: `git commit -m`, `git commit --amend`, `git push`, `git push --force`, `gh pr create` (which pushes), etc.

## Shell Command Style

- **Avoid compound `cd <path> && <command>` chains.** Claude Code's hardcoded compound-command protection prompts for approval on these even when both commands are individually allowed. Use alternatives that keep each tool call to a single command:
  - For git in another repo: use `git -C <path> <subcommand>` instead of `cd <path> && git <subcommand>`.
  - For non-git commands: pass the target path as an argument (e.g., `ls <path>`, `python <path>/script.py`) or use separate tool calls.
- Examples of read-only invocations that should not require approval: `git status`, `git diff`, `git log`, `git branch` (no flags), `git show`, `git stash list`, `git remote -v`, `git submodule status`, `git ls-files`, `git tag --list`. Filesystem reads (`ls`, `cat`) and benign local operations (`mkdir`) are also fine.
- Examples of invocations that always require explicit approval: `git commit`, `git push`, `git reset`, `git checkout`, `git rebase`, `git merge`, `git branch -d`, `git remote add/remove`, `git tag <name>` (creating/deleting), `git stash drop`.
- Filesystem commands like `cp` and `mv` are fine for scratch and temporary files. Moves or renames that affect git-tracked files should be reviewed before executing.
- **Avoid inline Python with `#` comments in quoted arguments.** Claude Code flags "newline followed by `#` inside a quoted argument" as a path-hiding risk and prompts for approval. Instead, write the code to a `.py` file and run `python <script>.py`.

## Environment Notes

- Prefer a Miniforge-managed Python interpreter. Miniforge ships both `conda` and `mamba`; prefer `mamba` for install and create operations (faster C++ solver) and fall back to `conda` only when a command is not supported by mamba (e.g., `conda rename`).
- If a `py312` environment or launcher exists, use it first.
- Do not conclude that Python is unavailable just because `python`, `python3`, or `py` fails in `PATH`; those may resolve to shims, store aliases, or the wrong interpreter.
- On Windows, a common Miniforge pattern is `%USERPROFILE%\\miniforge3\\envs\\py312\\python.exe`.
- On macOS or Linux, a common Miniforge pattern is `$HOME/miniforge3/envs/py312/bin/python`.
- If interpreter selection is still unclear, inspect Miniforge environments and local IDE settings before reporting that Python is missing.
- **PyCharm default interpreter:** The `py312` conda environment is configured as the default interpreter for new projects via **File > New Projects Setup > Settings for New Projects > Python Interpreter**. Existing cloned repos should also point to this environment unless they require a project-specific venv.
- GitHub CLI (`gh`) is used for PR and issue workflows. If `gh` is not found, remind the user to install it (`winget install GitHub.cli` on Windows, `brew install gh` on macOS) and authenticate with `gh auth login`.
- **Claude Code installation**: Prefer the **native installer**. Migrate off npm and winget when possible.
  - macOS: `curl -fsSL https://claude.ai/install.sh | sh`
  - Windows (PowerShell, no admin): `irm https://claude.ai/install.ps1 | iex` (requires Git for Windows)
  - To migrate from npm: `npm uninstall -g @anthropic-ai/claude-code` first. From winget: `winget uninstall Anthropic.ClaudeCode` first.
  - Native installs auto-update in the background by default. Use `/config` inside Claude Code to set the release channel (`latest` or `stable`). Run `claude doctor` to inspect updater status, and `claude update` to force an immediate update check.
  - To disable auto-updates, set `DISABLE_AUTOUPDATER=1` in the environment or add `"env": {"DISABLE_AUTOUPDATER": "1"}` to `~/.claude/settings.json`. Note: a legacy top-level `autoUpdates` key in `~/.claude.json` is ignored on native installs because `autoUpdatesProtectedForNative` neutralizes it.
- **Claude Code effort level**: The persisted `effortLevel` key in any `settings.json` only accepts `low`, `medium`, or `high`. Writing `"effortLevel": "max"` is silently discarded on read (the schema uses `.catch(undefined)`), so setting `max` that way is a no-op. To get `max` as a persistent default across every project and session, set the env var `CLAUDE_CODE_EFFORT_LEVEL=max`. The recommended place is `"env": {"CLAUDE_CODE_EFFORT_LEVEL": "max"}` inside `~/.claude/settings.json` (same mechanism used for `DISABLE_AUTOUPDATER`). The shared `user/settings.json` in this repo already sets this env entry, and bootstrap merges it into `~/.claude/settings.json`, so running bootstrap once on any consuming project lands the user-level default. Runtime precedence is managed policy > `CLAUDE_CODE_EFFORT_LEVEL` env var > persisted `effortLevel` in settings.json (resolved as local > project > user) > Claude Code's built-in default. The env var also outranks session-level controls. When `CLAUDE_CODE_EFFORT_LEVEL` is set, neither `--effort` at launch nor `/effort <level>` inside a session changes the current session, and the slash command prints a warning that the env var is overriding the live effort; `/effort low|medium|high|auto` still writes the persisted user setting, so that value takes effect once the env var is cleared, while `/effort max` has no lasting effect (`max` is not a valid persisted value). When the env var is unset, `--effort` at launch is a session-only override, `/effort low|medium|high|auto` updates the persisted user setting, and `/effort max` is session-only because `max` is not a valid persisted value.

## Submodule Workflow

- Some projects use git submodules for directories shared with collaborators (e.g., co-PI proposal repos, shared paper repos linked to Overleaf).
- At session start, if `.gitmodules` exists, run `git submodule status` to check submodule state. If submodules are uninitialized (prefix `-`), warn the user and suggest `git submodule update --init`.
- Submodule directories have their own `.git` and `origin` remote. Commits and pushes inside a submodule go to the submodule's upstream repo, not the parent.
- **Submodules are shared repos.** Pushes land directly in a collaborator's Overleaf project or co-PI repo. A careless force-push or overwrite can destroy someone else's work. Treat every write operation inside a submodule as high-risk.
- When the user asks to push or pull a submodule:
  1. Before writing, run `git -C <submodule-path> fetch` then `git -C <submodule-path> status` to check for uncommitted local changes. Review recent history with `git -C <submodule-path> log --oneline -5` to see local commits and `git -C <submodule-path> log --oneline -5 --remotes` to see recent remote-tracking activity. This is a quick sanity check, not a full divergence analysis; submodules are often in detached-HEAD state where branch comparisons do not apply cleanly.
  2. Use `git -C <submodule-path>` for git operations inside the submodule. Always confirm with the user before any commit, push, pull, or reset.
  3. Back in the parent repo, update the submodule pointer: `git add <submodule-path>` then commit (also requires confirmation).
- Submodules may have a `.gitignore` that excludes internal-only files (e.g., `.agent/`, `guardrail/`, `figure-spec/`, `figure-src/`). These files exist on disk but are not pushed to the collaborator repo. On a fresh clone, they will be missing. Warn the user if expected internal directories are absent.
- `context/` is synced to co-PI repos and will be available after submodule init.
- Project-specific submodule details (which directories, which upstream repos, which files are internal-only) belong in `CLAUDE.md` in each project repo, not here.

### Overleaf merge conflict resolution

Overleaf-synced repos (usually submodules) require special care during merges. Overleaf's git bridge creates branches from its own snapshot, which may lag behind the latest local push. When a collaborator edits on Overleaf while we push structural changes locally, the Overleaf branch is based on the **pre-push** state. In a merge, "theirs" means "older base plus collaborator styling edits," not "collaborator's newer version." Using `git checkout --theirs` on such files silently discards our work.

**Co-PI changes are the priority.** Our own structural work (compaction, renames) can be redone in minutes because we know exactly what we changed. A co-PI's content changes on Overleaf -- new sentences, rewritten arguments, added references, terminology choices -- represent their intellectual contribution. If we silently drop their edits, we may not even know what was lost, and they may not notice until weeks later. Losing their work is an order of magnitude worse than losing ours. The merge must preserve both sides, but when in doubt, err toward preserving the co-PI's content.

**Rules for merging Overleaf branches with conflicts:**

1. **Never use `git checkout --theirs`** on files where we have local structural changes (compaction, renames, reorganization). This is the single most dangerous command in an Overleaf merge.
2. **Never use `git checkout --ours` and stop there.** Starting from our version is correct, but the merge is not done until the co-PI's content changes are accounted for. Treating `--ours` as the final answer silently drops their work.
3. **Inspect what the collaborator actually changed** before resolving. First find the merge base: `git merge-base HEAD <overleaf-branch>`. Then run `git diff <merge-base>..<overleaf-branch> -- <file>` to isolate the co-PI's edits relative to the common ancestor, without mixing in our structural changes. Classify each change as:
   - **Content** (new sentences, rewritten arguments, added references, deliberate deletions or shortenings, terminology changes) -- must be preserved. Treat co-PI deletions with the same care as additions; if they removed text, that was a deliberate editorial decision, not noise.
   - **Formatting** (spacing, font commands, styling) -- apply if consistent with our version.
   - **Stale reversions** (undoes our rename or compaction because they edited the pre-push snapshot) -- discard, but note that the co-PI has not seen our change yet. Be careful: a change that looks like a stale reversion may actually be a deliberate content choice. When ambiguous, ask the user.
4. **Apply their content changes onto our structural base.** Start from `git checkout --ours <file>`, then manually integrate every content change identified in step 3. Do not skip any co-PI content change without explicit user approval.
5. **Double-verify before committing** -- check both directions:
   - `git diff <pre-merge-commit> -- <file>` -- confirm our structural changes survived.
   - `git diff <overleaf-branch> -- <file>` -- confirm the only differences from the co-PI's version are our intended structural changes, not dropped content.
   - If the co-PI added entirely new paragraphs or sections, verify they appear in the merged file.
6. **Screen for binary artifacts before staging.** Overleaf branches often carry compiled PDFs, review screenshots (`out-review/`), or other build artifacts that should not be tracked. Use the same merge-base diff from pre-merge checklist step 2 to spot unexpected large files. Add them to `.gitignore` before staging the merge.

**Pre-merge checklist** (run before `git merge <overleaf-branch>`):

1. `git fetch` to get the latest Overleaf branch.
2. `git diff --stat $(git merge-base HEAD <overleaf-branch>)..<overleaf-branch>` -- check which files the co-PI actually touched, spot binary artifacts.
3. `git log --oneline HEAD..<overleaf-branch>` -- understand what the collaborator did.
4. If any files we modified structurally appear in the diff, plan to resolve those conflicts manually using the rules above.
5. If the co-PI touched files we did not modify, those should auto-merge cleanly. After the merge, still spot-check them for unintended content loss.

**Recovery if `--theirs` was already used:** Restore our structural version from the pre-merge commit with `git restore --source=<pre-merge-commit> --worktree -- <file>` (avoids encoding and line-ending issues from shell redirection on Windows). Then reapply the collaborator's content and formatting changes on top. Do not skip the reapply step -- their work matters too.

## Local Skills Precedence

- If the workspace contains a `skills/` directory, treat repo-local skills as the default source of truth for that project.
- When a task matches a skill name and both a repo-local `skills/<skill-name>/SKILL.md` and an installed global skill exist, prefer the repo-local skill.
- When using a repo-local skill, read `skills/<skill-name>/SKILL.md` and its local `references/`, `scripts/`, and `assets/` before falling back to any globally installed copy.
- Do not modify a globally installed skill when a repo-local skill of the same name exists, unless the user explicitly asks to update the global copy too.
- If a repo-local skill overrides a global skill, state briefly that the local project copy is being used.

## Cross-Tool Skill Sharing

- Skills under `skills/` are shared between coding agents (Codex, Claude Code, and any future agent).
- `skills/<skill-name>/SKILL.md` is the single source of truth for each skill. Agent-specific config files (e.g., `agents/openai.yaml`) are thin wrappers and must not duplicate or override the logic in `SKILL.md`.
- Claude Code accesses these skills via pointer commands in `.claude/commands/`. Each pointer file references the corresponding `SKILL.md` rather than duplicating its content.
- Bootstrap sync should copy only the shared repo's `.claude/commands/*.md` files into the project `.claude/commands/` directory and should not delete unrelated project-local commands.
- When editing a skill, modify `SKILL.md` and its `references/` or `scripts/` directly. Do not create agent-specific forks of the same content.
- If a new skill is added, create both the `skills/<skill-name>/SKILL.md` structure and a matching `.claude/commands/<skill-name>.md` pointer so both agents can use it immediately.
