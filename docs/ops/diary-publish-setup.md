# Diary Publish — Deploy Key Setup

## 1. Generate deploy key (on Jetson)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/diary_publish_ed25519 -N "" -C "reachy-diary-publish"
chmod 600 ~/.ssh/diary_publish_ed25519
```

Copy the public key (`~/.ssh/diary_publish_ed25519.pub`) and add it to the
site repo's GitHub Settings → Deploy keys, with **Write access enabled**.

## 2. SSH host alias

Add to `~/.ssh/config`:

```
Host github-diary-site
  HostName github.com
  User git
  IdentityFile ~/.ssh/diary_publish_ed25519
  IdentitiesOnly yes
```

## 3. Environment variables for the OpenClaw skill

```
SITE_REPO_URL=git@github-diary-site:<owner>/<site-repo>.git
SITE_DIARY_PATH=content/<diary section>      # supplied by user; e.g. content/journal
SITE_BRANCH=main
GIT_AUTHOR_NAME="Reachy Mini"
GIT_AUTHOR_EMAIL="reachy@local"
GIT_COMMITTER_NAME="Reachy Mini"
GIT_COMMITTER_EMAIL="reachy@local"
```

## 4. First run

```bash
uv run python scripts/publish_diary.py --date 2026-04-26
```

If the clone succeeds and the push lands, you're done.

## 5. Key rotation

To rotate: generate a new key, add it to GitHub deploy keys, swap the
`IdentityFile` path, then remove the old key from GitHub.

## 6. OpenClaw skill

The `daily-diary` OpenClaw skill (in `~/project/openclaw/extensions/desktop-robot/src/`) is responsible for triggering generation + publish at 23:00 daily. Skill flow:

1. `uv run python scripts/generate_diary.py --date $(date +%F)` — fail-fast on non-zero
2. `uv run python scripts/publish_diary.py --date $(date +%F)` — fail-fast on non-zero

The skill definition and cron trigger live in the OpenClaw repo and are committed there separately.