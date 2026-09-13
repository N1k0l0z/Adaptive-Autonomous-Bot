import sys
import subprocess
import torch

DOCUMENT_UPLOAD_SERVICE = "document_upload_service"
MAIN_SERVICES = ["database_pgvector", "embedding_service", "api_gateway"]


def check_cuda() -> bool:
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available on this device.")
    return has_cuda


def ask_user_choice() -> bool:
    while True:
        choice = input("Do you want to run document upload? [y/n]: ").strip().lower()
        if choice in ["y", "yes"]:
            return True
        if choice in ["n", "no"]:
            return False
        print("Invalid input. Type 'y' or 'n'.")


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main():
    print("--- System Startup ---")
    cuda_ready = check_cuda()
    should_upload = False

    if not cuda_ready:
        print("Document upload requires CUDA. Skipping upload option.")
    else:
        should_upload = ask_user_choice()

    if should_upload:
        print(f"\n🚀 Building and running {DOCUMENT_UPLOAD_SERVICE}...")
        run_cmd(["docker", "compose", "build", DOCUMENT_UPLOAD_SERVICE])
        run_cmd(["docker", "compose", "run", "--rm", DOCUMENT_UPLOAD_SERVICE])
        print("Document upload completed.")

    print("\nStarting main services...")
    for service in MAIN_SERVICES:
        print(f"🚀 Starting {service}...")
        run_cmd(["docker", "compose", "up", "--build", "-d", service])

    print("\nSystem startup finished!")


if __name__ == "__main__":
    main()