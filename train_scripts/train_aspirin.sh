# Run all aspirin configs sequentially
CONFIG_DIR="configs/aspirin"

for config in "$CONFIG_DIR"/*.yml; do
    echo "Running: $config"
    python parallel_lightning.py --config "$config"
    echo "Finished: $config"
done

echo "All jobs completed!"

