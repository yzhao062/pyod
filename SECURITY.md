# Security Policy

PyOD is used across production systems for anomaly detection. We treat security reports
with priority and handle them through coordinated disclosure.

## Supported Versions

Security fixes land on the latest released version of PyOD. Please upgrade to the current
release before reporting, in case the issue is already resolved.

## Reporting a Vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability reporting.
Do not open a public issue for a security problem.

Under the **Security** tab of this repository, select **Report a vulnerability**.

Include the affected version, a description of the problem, the impact, and a minimal
reproduction or proof of concept if you have one.

## What to Expect

- We aim to acknowledge a report within three business days.
- We will confirm the issue, assess its severity, and keep you updated while we prepare a
  fix.
- Please keep the report private while we investigate. We aim to remediate within 90 days
  and will coordinate the disclosure date with you; any extension will be agreed together.
- With your consent, we credit reporters in the release notes and the published advisory.

## Scope

In scope: the PyOD source code, including the model and artifact loading paths,
deserialization (for example ``pyod.utils.persistence``), and release integrity.

Out of scope: vulnerabilities in third-party dependencies (report those upstream), issues
that require an already-compromised local environment, and findings against unsupported or
modified versions.
