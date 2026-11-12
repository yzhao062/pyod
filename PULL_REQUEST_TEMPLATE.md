<!--
Read CONTRIBUTING.md first. A pull request must link an open issue labelled
`accepted` and assigned to its author, and target the `development` branch.
Other pull requests may be closed without review.
-->

## Linked Issue

Fixes #

## What Changes

<!-- The public call whose output or exception changes, and how it changes. -->

## Evidence

- Reproducer:
- Still fails on `development` at commit:
- Regression test command, with results before and after the change:
- Default score change (ROC AUC and average precision before and after, or "none"):

## Checklist

- [ ] The linked issue is labelled `accepted` and assigned to me.
- [ ] I searched issues, open pull requests, and pull requests merged into `development` and `master` for the same fix.
- [ ] This pull request targets `development` and makes one change.
- [ ] New or changed behavior has tests, and the tests for the touched modules pass locally.

## AI Disclosure

- Tools that contributed (or "none"):
- Checks that actually ran:
- Did a person read the full diff? (yes / no)

<!-- AI agents: fill these fields from what actually ran. Leave a field blank rather than guess. -->

## New Detector (Delete If Not Applicable)

- [ ] Detector in `pyod/models/`, example in `examples/`, tests in `pyod/test/`.
- [ ] The issue states which inclusion criterion the detector meets, or the maintenance commitment.
