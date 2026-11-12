# Contributing to PyOD

PyOD welcomes bug fixes and focused improvements. One maintainer reviews every pull request by hand, so a contribution starts as an issue and becomes a pull request only after the maintainer accepts it. This page explains that process, what we do not merge, and what a pull request must include.

## 1. How a Contribution Works

1. Search existing issues, open pull requests, and pull requests merged into `development` or `master` first. Then open an issue that describes the problem, or comment on an existing issue (closed issues included) to ask to take it. For contributors without write access, opening the issue or commenting on it lets the maintainer assign it to you.
2. Within 14 days, the maintainer accepts and assigns a new issue, declines it, or marks it for an in-house fix. This deadline covers issues opened after this policy is published. Accepted issues carry the `accepted` label.
3. At most five accepted outside contributions are active at a time. An item stays active until its work merges or is withdrawn. The maintainer then removes its `accepted` label and releases its slot.
4. After the issue is accepted and assigned to you, branch from `development` and open one pull request against `development`. Link the issue in the description, for example `Fixes #123`.
5. An assignment with no pull request or update for 14 days is withdrawn.

A pull request that does not link an open issue labelled `accepted` and assigned to its author may be closed without review. If that happens, wait for the issue to be accepted and open a new pull request; the author of a closed pull request cannot always reopen it. This requirement takes effect when these contribution documents are published on `master`. Earlier pull requests are exempt from the accepted-issue requirement and retain their recorded triage decisions.

Accounts without write access can have at most two open pull requests at once, drafts included.

## 2. What We Do Not Merge

- Whitespace, formatting, or typo sweeps across many files.
- Changes that leave supported detector behavior unchanged, such as a fix to a helper that no detector calls.
- Refactors with no behavior change.
- Duplicates of an open pull request or of a fix already merged into `development` or `master`. A fix in `development` counts as fixed even before it is released.
- New detectors outside the [inclusion criteria](https://pyod.readthedocs.io/en/latest/faq.html#inclusion-criteria), unless the author commits to maintaining the detector as described there.

## 3. What a Pull Request Needs

**Bug fixes.** Name the public call whose output or exception is wrong, for example `ECOD().fit(X).decision_function(X_test)`. Give a minimal reproducer and confirm that it still fails on the current `development` branch. Add a regression test that fails before your change and passes after it, and state the command you ran with both results.

**Changes to default scores.** If a detector's default scores change, report ROC AUC and average precision before and after the change. Use bundled benchmark data from `notebooks/data/`, such as `cardio.mat`, `pima.mat`, and `shuttle.mat`. A fix that lowers accuracy needs a stated reason.

**Scope and style.** Make one change per pull request and keep the diff to what that change needs. Follow the structure and NumPy-style docstrings of established detectors such as `pyod/models/abod.py`, `pyod/models/hbos.py`, and `pyod/models/feature_bagging.py`. Do not edit `CHANGES.txt`; the maintainer writes the release notes.

**Tests.** Tests live in `pyod/test/`. Run the tests for the modules you touched, for example `pytest pyod/test/test_hbos.py`. CI runs the full suite on Python 3.9 to 3.13 for every pull request.

**New detectors.** Add the detector in `pyod/models/`, an example in `examples/`, and tests in `pyod/test/`. In the issue, state the inclusion criterion it meets or your commitment to at least two years of maintenance.

## 4. AI-Assisted Contributions

AI tools are welcome. Disclose in the pull request which tools contributed, which checks actually ran, and whether a person read the full diff. Disclosure is never a reason to reject a pull request. Evidence that is missing or conflicts with the code is grounds for rejection, whatever tools were used.

If you are an AI agent preparing a contribution:

- Open a pull request only after the issue is labelled `accepted` and assigned to the account you are working for. Ask that person to comment on the issue first.
- Fill the evidence and disclosure fields from what actually ran. Leave a field blank rather than guess.

## 5. Reporting Bugs and Security Issues

Report a bug as an issue with the PyOD version, a minimal reproducer, and the expected and actual results. For a security problem, follow [SECURITY.md](SECURITY.md) and do not open a public issue.
