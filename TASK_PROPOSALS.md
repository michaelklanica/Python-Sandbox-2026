# Codebase Task Proposals

## 1) Typo fix task
- **Issue:** The output string in `helloworld.py` uses `Hello, World!` (capitalized phrase) while the filename and typical script naming imply a canonical lowercase `hello world` sample. This is minor copy inconsistency.
- **Task:** Normalize the greeting string style to match the project convention (for example, `Hello world!` or update file naming/content so wording is intentional and consistent).
- **Why:** Keeps introductory/sample script messaging consistent and intentional.

## 2) Bug fix task
- **Issue:** `_toggle_sign` only works when the entire expression is a plain number. If a user enters a normal expression like `12+3`, pressing `±` sets `result_var` to `Invalid` instead of toggling the current operand or failing gracefully.
- **Task:** Update sign-toggle behavior to operate on the active operand (e.g., convert `12+3` to `12+(-3)`) or disable `±` unless input is a standalone numeric value.
- **Why:** Current behavior surprises users and introduces an error state for common calculator workflows.

## 3) Comment/docs discrepancy task
- **Issue:** The class docstring says the app is a "modern desktop calculator," but `_evaluate` uses `eval`, replacing `%` with `/100`, which differs from many calculator `%` semantics (e.g., percentage of previous operand) and has limited expression parsing.
- **Task:** Clarify documentation around supported expression syntax and `%` behavior, or implement behavior consistent with standard calculator expectations.
- **Why:** Aligns user-facing documentation with actual behavior and avoids confusion.

## 4) Test improvement task
- **Issue:** The repository has no automated tests for calculator logic (`_evaluate`, `_toggle_sign`, `_backspace`, error handling).
- **Task:** Add focused unit tests that instantiate `CalculatorApp` with a Tk root in headless mode and validate:
  - basic arithmetic (`2+2 -> 4`)
  - float normalization (`4/2 -> 2`, `3/2 -> 1.5`)
  - invalid expressions return `Error`
  - sign toggle behavior on standalone numbers and composed expressions
  - `%` semantics expected by documentation
- **Why:** Prevents regressions and documents intended behavior.
