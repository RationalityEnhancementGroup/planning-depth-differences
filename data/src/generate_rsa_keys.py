"""Generates RSA keys for encrypting worker IDs"""
from argparse import ArgumentParser
from pathlib import Path

import rsa

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        dest="key_name",
    )

    inputs = parser.parse_args()
    data_path = Path(__file__).resolve().parents[1]

    public_key, private_key = rsa.newkeys(512)

    data_path.joinpath(data_path.joinpath("inputs/keys")).mkdir(
        parents=True, exist_ok=True
    )

    with open(data_path.joinpath(f"inputs/keys/pubkey_{inputs.key_name}"), "wb") as f:
        f.write(public_key.save_pkcs1())

    with open(data_path.joinpath(f"inputs/keys/priv_{inputs.key_name}"), "wb") as f:
        f.write(private_key.save_pkcs1())
