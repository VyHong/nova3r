find 3D-FRONT-TEST-SCENE -name "*.ply" -type f | while read -r ply_file; do
   # 1. Check if the file is completely empty (This one caught a real empty file in your log!)
    if [ ! -s "$ply_file" ]; then
        echo "[EMPTY] $ply_file"
        continue
    fi

    # 2. Extract vertex count using -a (treat binary as text) and suppress grep errors
    num_vertices=$(grep -a -m 1 "element vertex" "$ply_file" 2>/dev/null | awk '{print $3}')

    if [ -z "$num_vertices" ]; then
        echo "[REAL BAD HEADER] $ply_file (Missing 'element vertex' specification)"
        continue
    fi

    # 3. Check format using -a
    format=$(grep -a -m 1 "format" "$ply_file" 2>/dev/null | awk '{print $2}')

    if [ "$format" = "ascii" ]; then
        actual_lines=$(sed '1,/end_header/d' "$ply_file" | wc -l)
        if [ "$actual_lines" -lt "$num_vertices" ]; then
            echo "[CORRUPTED ASCII] $ply_file (Expected $num_vertices, found $actual_lines)"
        fi
    else
        # 4. Binary File Validation
        # A 'binary_little_endian' PLY file has a tiny text header, then raw bytes.
        # Each vertex (double precision X, Y, Z) takes exactly 24 bytes (3 * 8 bytes).
        expected_min_bytes=$((num_vertices * 24))
        actual_bytes=$(stat -c%s "$ply_file")
        
        if [ "$actual_bytes" -lt "$expected_min_bytes" ]; then
             echo "[CORRUPTED BINARY] $ply_file (File size $actual_bytes bytes is too small for $num_vertices vertices)"
        fi
    fi
done