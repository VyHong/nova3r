#!/bin/bash

# Usage: ./bulk_mount.sh /path/to/zips /path/to/output
SRC_DIR=$1
DEST_DIR=$2

# Check if arguments are provided
if [ -z "$SRC_DIR" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: $0 <source_folder_with_zips> <target_mount_folder>"
    exit 1
fi

# Create the destination root if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Starting mount process..."
echo "--------------------------"

for zip_path in "$SRC_DIR"/*.zip; do
    # 1. Get the filename without the .zip extension
    zip_name=$(basename "$zip_path" .zip)
    
    # 2. Create a dedicated mount point for this specific zip
    mount_point="$DEST_DIR/$zip_name"
    mkdir -p "$mount_point"

    # 3. Perform the mount using the Google mount-zip syntax
    # -o notrim ensures the internal folder structure is preserved
    echo "Mounting: $zip_path"
    echo "Mount point: $mount_point"
    fuse-zip -o nonempty "$zip_path" "$mount_point"
    #fusermount -u "$mount_point" 

    if [ $? -eq 0 ]; then
        echo "✅ Mounted: $zip_name"
    else
        echo "❌ Failed to mount: $zip_name"
    fi
done

echo "--------------------------"
echo "All done! Mounts are available in $DEST_DIR"