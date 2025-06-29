#!/bin/bash
# Final fix for all datetime/timezone import issues
# Save as: fix_all_timezone.sh and run: bash fix_all_timezone.sh

echo "Fixing all timezone imports..."

# List of all Python files that might use datetime
files=(
    "infrastructure/processors/data_processor.py"
    "infrastructure/collectors/websocket.py"
    "infrastructure/collectors/rest.py"
    "infrastructure/database/manager.py"
    "infrastructure/scheduler/tasks.py"
    "infrastructure/message_bus/bus.py"
    "scripts/run_processor.py"
    "scripts/run_collector.py"
    "scripts/test_infrastructure.py"
    "scripts/migrate_data.py"
    "core/models.py"
)

# Fix each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        # Check if file uses timezone
        if grep -q "timezone.utc\|timezone(" "$file"; then
            # Check if timezone is imported
            if ! grep -q "from datetime import.*timezone" "$file"; then
                echo "Fixing $file..."
                # Add timezone to datetime imports
                sed -i '' 's/from datetime import \(.*\)$/from datetime import \1, timezone/' "$file"
                # Remove duplicate timezone if added twice
                sed -i '' 's/, timezone, timezone/, timezone/' "$file"
            fi
        fi
        
        # Fix deprecated utcnow()
        if grep -q "datetime.utcnow()" "$file"; then
            echo "Fixing utcnow() in $file..."
            sed -i '' 's/datetime.utcnow()/datetime.now(timezone.utc)/g' "$file"
        fi
    fi
done

echo "✅ All timezone imports fixed!"
echo ""
echo "Now try running the processor again:"
echo "python scripts/run_processor.py"