import os
import sys

def collect_txt_file_names(path):
    """Recursively collect all .txt file paths under the given directory."""
    txt_file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_file_names.append(file)
    return txt_file_names

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3 or sys.argv[1] != "--path":
        print("Usage: python collect_book.py --path <directory_path> [--output <output_file>]")
        print("Example: python collect_book.py --path /home/user/books --output txt_files_list.txt")
        sys.exit(1)

    # Parse --path
    if sys.argv[1] == "--path":
        directory_path = sys.argv[2]
    else:
        print("Error: --path argument is required.")
        sys.exit(1)

    # Check for optional --output
    output_file = None
    if "--output" in sys.argv:
        output_index = sys.argv.index("--output")
        if output_index + 1 >= len(sys.argv):
            print("Error: --output requires a file path.")
            sys.exit(1)
        output_file = sys.argv[output_index + 1]

    # Validate directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)

    # Collect .txt files
    txt_files = collect_txt_file_names(directory_path)

    # Output results
    if txt_files:
        # Save to file if --output is provided
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    for file_path in txt_files:
                        f.write(file_path + "\n")
                print(f"Found {len(txt_files)} .txt file(s). List saved to: {output_file}")
            except Exception as e:
                print(f"Error writing to output file: {e}")
                sys.exit(1)
        else:
            # Print to console if no output file specified
            for file_path in txt_files:
                print(file_path)
            print(f"\nTotal: {len(txt_files)} .txt file(s) found.")
    else:
        message = "No .txt files found in the specified directory."
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(message + "\n")
            print(message)
            print(f"Empty result saved to: {output_file}")
        else:
            print(message)