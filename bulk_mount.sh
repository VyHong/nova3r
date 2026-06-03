#!/bin/bash

# Usage: ./bulk_mount.sh <path_to_zip_or_dir> /path/to/output
SRC_TARGET=$1
DEST_DIR=$2

# Check if arguments are provided
if [ -z "$SRC_TARGET" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: $0 <source_folder_or_single_zip> <target_mount_folder>"
    exit 1
fi

# Create the destination root if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Starting mount process..."
echo "--------------------------"

# Function to perform the actual mount
mount_single_zip() {
    local zip_path="$1"
    
    # 1. Get the filename without the .zip extension
    local zip_name=$(basename "$zip_path" .zip)
    
    # 2. Create a dedicated mount point for this specific zip
    local mount_point="$DEST_DIR/$zip_name"
    mkdir -p "$mount_point"

    # 3. Perform the mount
    echo "Mounting: $zip_path"
    echo "Mount point: $mount_point"
    fuse-zip -o nonempty "$zip_path" "$mount_point"

    if [ $? -eq 0 ]; then
        echo "✅ Mounted: $zip_name"
    else
        echo "❌ Failed to mount: $zip_name"
    fi
}

# Check if the input is a single file or a directory
if [ -f "$SRC_TARGET" ]; then
    # It's a single file, make sure it's a zip
    if [[ "$SRC_TARGET" == *.zip ]]; then
        mount_single_zip "$SRC_TARGET"
    else
        echo "❌ Error: File is not a .zip file"
        exit 1
    fi
elif [ -d "$SRC_TARGET" ]; then
    # It's a directory, loop through all zips inside
    for zip_path in "$SRC_TARGET"/*.zip; do
        # Guard against the case where no .zip files exist in the directory
        [ -e "$zip_path" ] || continue
        mount_single_zip "$zip_path"
    done
else
    echo "❌ Error: $SRC_TARGET is neither a valid file nor directory"
    exit 1
fi

echo "--------------------------"
echo "All done! Mounts are available in $DEST_DIR"