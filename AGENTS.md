## ALWAYS RUN

Important: this list is separate from any others in the prompt. Follow this
order of operations.

1. study the README.md
2. grep `specs/` for any relevant keywords to the task you are implementing.
3. study the relevant specs.
4. grep `design/` for any relevant keywords to the task you are implementing.
5. study the relevant documents.
6. implement the change requested in the prompt.
7. run linting and formatting before committing.
8. Identify any remaining issues or features that need to be implemented
   1. file them as bd issues (see [Issue Management](#issue-management)).
   2. include them in GAPS.md
9. commit and push the change.
10. if you were not able to fully address the bd issue, mark it as "blocked".
    This will ensure the next worker does not pick it up again until a human reviews.

## Branch cleanup

When starting work, the branch should be clean, and you should try pull the
latest changes from the primary upstream branch before continuing.

## Add and update documentation

Always add and update documentation as appropriate. Update at least the following:

- any relevant files in the `docs/` directory.
- any updated designs and considerations in the `specs/` directory.

## Issue Management

File issues liberally to help keep context minimal and focus on your current
task.

- Issues are managed via the `bd` commmand line.
- Issues are created via `bd create`
  - Priority should be set to the following (P2 otherwise):
    - P0: Any required project fundamentals or initialization

## Committing code

- **unless** the prompt contains "don't commit", commit the code.
- **unless** the prompt contains "don't push", push the code.

- Use the conventional commit format for commit messages.
- The commit description must explain the problem first.
- The commit description must a summary of each area modified.

## Linting

- Always run linting and formatting before committing.
- Formatting and lint fixing tools are available via `just fix`.
- Linting tools are available via `just lint`.

## Testing

- linting, formatting, and testing **must** pass before a commit.
- add unit tests for every change if possible.

## CI

CI **must** pass after every commit.

To verify CI status, use the GitHub MCP server.

## Code Design

The following code tenants are followed:

- functional programming as much as possible.
- separate state from functional programming.
- re-use code as much as possible.
- leverage best-practice third party libraries.

## Examples

- example data is in the `examples/` directory.
