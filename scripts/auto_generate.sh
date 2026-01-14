SCRIPTS=(
    "scripts/insert_pipeline_methods.py"
    "scripts/insert_group_by_methods.py"
    "scripts/insert_gbdt_methods.py"
    "scripts/insert_metrics_methods.py"
    "scripts/insert_optimize_methods.py"
    "scripts/insert_horizontal_methods.py"
    "scripts/insert_feature_engineering_methods.py"
    "scripts/insert_unified_io_methods.py"
)

for script in "${SCRIPTS[@]}"; do
    echo "Running $script..."
    uv run python "$script"
done

echo "Formatting code..."
uv format

echo "Done!"
