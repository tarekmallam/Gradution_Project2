# Graduation Project 2

## Troubleshooting

- **Database connection failed:**  
  ```
  ❌ Connection failed to the database: could not translate host name ...
  ```
  - Check your internet connection.
  - Verify your database host and credentials in your configuration files.

- **Model file is corrupted:**  
  ```
  OSError: Unable to synchronously open file (truncated file: ...)
  ```
  - Re-upload or re-download the `model.h5` file to your project directory.
  - Ensure the file is not corrupted or incomplete.
  - The file size should match the original model export.

  **How to fix:**
  1. Obtain a valid, complete `model.h5` file (from your training/export process or your team).
  2. Replace the corrupted `model.h5` in your project directory with the correct one.
  3. Make sure the file is fully uploaded and not interrupted.

  **You cannot fix this error in code.**  
  It is a data/file issue, not a programming bug.

- **General tip:**  
  Always use `python main.py` to run your project, not `source main.py`.

## Model File

- Ensure `model.h5` is the correct, complete model file.
- Place it in the required directory (e.g., `/workspaces/Gradution_Project2/website/model.h5`).
- If you replace the file, re-run your application with `python main.py`.