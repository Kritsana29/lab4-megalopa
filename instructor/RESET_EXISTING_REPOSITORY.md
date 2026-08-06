# Replace the Existing Central-Branch Repository with the Fork Workflow

This procedure keeps the existing GitHub repository but replaces its tracked contents.

## 1. Back up anything needed

Download or copy any dataset or validation files that must be retained.

## 2. In the existing local repository

```bash
git checkout main
git pull --ff-only origin main
```

Remove all currently tracked files while preserving `.git/`:

```bash
git rm -r .
```

Copy the extracted contents of `Lab4_Megalopa_Fork_Workflow_v4/` into the repository root. Confirm the root directly contains:

```text
README.md
START_HERE.md
shared/
student_work/
instructor/
```

Then stage additions and deletions:

```bash
git add -A
git status
git commit -m "Replace group-branch workflow with student fork workflow"
git push origin main
```

## 3. Delete obsolete group branches

Only do this after confirming the fork workflow is final and no student work must be retained:

```bash
for SECTION in 01 02; do
  for GROUP in $(seq -w 1 25); do
    git push origin --delete "sec${SECTION}-group${GROUP}"
  done
done
```

## 4. Remove obsolete GitHub rules

In GitHub repository settings:

- Remove any group-branch rulesets or group-folder workflow rules.
- Keep `main` protected for instructor maintenance.
- Students do not need repository collaborator access to the upstream repository.

## 5. Add the real data and validation material

After inserting the real files:

```bash
git add shared/dataset shared/validation
git commit -m "Add final Lab 4 dataset and validation materials"
git push origin main
```
