from operation_agent.execution import ensure_persistent_container
import subprocess

print("--- 1st Run (Should create new) ---")
try:
    cname = ensure_persistent_container(container_name="test_automl_worker", device="0")
    print(f"Result: {cname}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- 2nd Run (Should reuse existing) ---")
try:
    cname = ensure_persistent_container(container_name="test_automl_worker", device="0")
    print(f"Result: {cname}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Cleanup ---")
subprocess.run(["docker", "rm", "-f", "test_automl_worker"])
print("Cleanup done")
