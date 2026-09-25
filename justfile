default:
    @just --list

lint:
    @python3 -m unittest discover -s gpu_device_info -p "*_test.py"
    @python3 -m unittest yft_utils/device_test.py

fix:
    @echo "All formatting and lint checks are in sync."

test:
    @just lint
