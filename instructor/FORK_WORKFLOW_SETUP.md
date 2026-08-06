# Instructor Setup - Fork Workflow

## Repository policy

- Keep `WUStuLab/lab4-megalopa` public and instructor-controlled.
- Students do not receive Write access to the upstream repository.
- Protect upstream `main` from direct student changes.
- Every student must have a GitHub account.
- One student per group creates the fork; all remaining group members are added as collaborators.

## Before Session 1

1. Finalize the dataset, validation images, and validation video.
2. Verify the class mapping in the Session 1 notebook.
3. Test one complete temporary fork.
4. Freeze the upstream repository.
5. Ask groups to submit their fork URLs before training begins.

## Student token

For the final Session 1 push, the GitHub account used to push should create a fine-grained token limited to the group fork with:

- Repository access: only the group fork
- Repository permission: Contents - Read and write

The token is stored in Colab Secrets as `GITHUB_TOKEN` and must not appear in the PDF.

## Session 2

The Pi kit only clones the public group fork. It does not need a token because no Session 2 push is required.
