import re
import os
from pathlib import Path


def sync_readme(readme_path):
    """
    Reads a README file, looks for <!-- include: filepath --> markers,
    and updates the following code block with the content of the file.
    """
    try:
        with open(readme_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {readme_path}")
        return

    # Regex to find the marker and the following code block
    # Matches:
    # 1. <!-- include: path/to/file --> (with optional whitespace)
    # 2. The opening fence of the code block (e.g. ```python)
    # 3. The content inside the code block (non-greedy)
    # 4. The closing fence (```)
    pattern = re.compile(r"(<!--\s*include:\s*(.+?)\s*-->\n)(```\w*\n)([\s\S]*?)(```)", re.MULTILINE)

    def replace_chunk(match):
        marker_line = match.group(1)
        rel_path = match.group(2).strip()
        opening_fence = match.group(3)
        # old_content = match.group(4)
        closing_fence = match.group(5)

        # Resolve path relative to README location
        readme_dir = os.path.dirname(readme_path)
        source_path = os.path.join(readme_dir, rel_path)

        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path} (referenced in {readme_path})")
            return match.group(0)  # Return original text if file not found

        try:
            with open(source_path, "r") as src:
                new_content = src.read()
                # Ensure the content ends with exactly one newline if it's not empty
                new_content = new_content.rstrip() + "\n"
        except Exception as e:
            print(f"Error reading {source_path}: {e}")
            return match.group(0)

        return f"{marker_line}{opening_fence}{new_content}{closing_fence}"

    new_content = pattern.sub(replace_chunk, content)

    if new_content != content:
        with open(readme_path, "w") as f:
            f.write(new_content)
        print(f"Updated {readme_path}")
    else:
        print(f"No changes needed for {readme_path}")


if __name__ == "__main__":
    # Find all README.md in examples/
    # You can adjust the root path as needed
    root = Path("examples")
    if not root.exists():
        # Fallback if running from a different directory
        root = Path(".")

    print("Scanning for README.md files...")
    for readme in root.rglob("README.md"):
        sync_readme(readme)
