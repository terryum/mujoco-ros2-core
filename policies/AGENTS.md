# Robotics workspace instructions

## Repository visibility

- Use the `terryum` GitHub account and `terry.t.um@gmail.com` commit email for
  every repository under `~/Codes/robotics`.
- Repositories containing Friday-specific source, model adapters, configuration,
  motion, learning data, or hardware integration must be private.
- Repositories without Friday-specific tracked content must be public and use
  Apache-2.0 by default. This includes common tooling, Wuji Hand 2, Franka, and
  other public robot-model experiments.
- If a project mixes Friday and non-Friday work, split out the generic/public
  portion. If it cannot be split, the Friday private rule takes precedence.
- A private Friday repository name may be mentioned publicly, but its private
  implementation and data must not be copied into public history.

## GitHub routing

- Run `gh context` before repository creation, rename, visibility changes, or
  other consequential GitHub writes. It must report the personal `terryum`
  account and owner.
- Never use `terry-cosmax` for repository creation, fork, push, or pull request
  work. It is only for read-only access to approved `Holiday-Robot/*` upstreams.
- Never run `gh auth switch`; account selection is directory-routed.
- Keep every repository as a direct child of `~/Codes/robotics` with a matching
  local folder name and GitHub repository name. The workspace root is not a Git
  repository.

## Assets, data, and hardware

- Preserve vendor repositories as pinned, read-only dependencies. Do not modify
  or push upstream assets without an explicit request.
- When redistribution is not permitted, publish only a download/source manifest,
  version, checksum, and adapter code.
- Keep credentials, serial numbers, private IPs, site calibration, raw datasets,
  bags, runs, and checkpoints out of Git.
- New hardware adapters start read-only. Require model identity, safety state,
  limits, acknowledgement, timeout, and stale-command checks before motion.

