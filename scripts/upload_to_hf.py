"""Script para subir el modelo Laya Service Desk y su Space interactivo a Hugging Face Hub.

Lee automáticamente el token desde el archivo .env en la raíz del proyecto.
"""

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi


def get_hf_token() -> str:
    """Carga el token de Hugging Face desde el archivo .env."""
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        # Fallback de lectura directa
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "HUGGINGFACE_TOKEN" in line:
                        token = line.split("=", 1)[1].strip().strip("\"' ")
                        break

    if not token:
        raise ValueError("No se encontró HUGGINGFACE_TOKEN en el archivo .env")

    return token


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Subir modelo y Space a Hugging Face")
    parser.add_argument("--model-repo", type=str, default=None, help="Nombre del repo del modelo (ej: JuanRetamales/laya-service-desk-es)")
    parser.add_argument("--space-repo", type=str, default=None, help="Nombre del Space demo (ej: JuanRetamales/laya-service-desk-demo)")
    parser.add_argument("--skip-model", action="store_true", help="Omitir la subida de los pesos del modelo")
    parser.add_argument("--skip-space", action="store_true", help="Omitir la subida de la demo Space")

    args = parser.parse_args()

    token = get_hf_token()
    api = HfApi(token=token)

    # Validar usuario
    user_info = api.whoami()
    username = user_info.get("name", "JuanRetamales")
    print(f"🔑 Autenticado en Hugging Face como: {username} ({user_info.get('fullname')})")

    model_repo_id = args.model_repo or f"{username}/laya-service-desk-es"
    space_repo_id = args.space_repo or f"{username}/laya-service-desk-demo"

    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints" / "laya_service_desk"
    space_dir = root_dir / "space"

    # 1. Subir Modelo
    if not args.skip_model:
        if not checkpoints_dir.exists():
            print(f"❌ Error: La carpeta de checkpoints no existe en {checkpoints_dir}")
            sys.exit(1)

        print(f"\n📦 [1/2] Creando/Verificando repositorio de Modelo: {model_repo_id}...")
        api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)

        print(f"🚀 Subiendo archivos del modelo desde {checkpoints_dir} (~1.28 GB)...")
        print("   (Esto puede tomar unos minutos dependiendo de la velocidad de subida)")
        api.upload_folder(
            folder_path=str(checkpoints_dir),
            repo_id=model_repo_id,
            repo_type="model",
            ignore_patterns=["last_model.safetensors", "figures/*", "*.png"],
        )
        print(f"✅ ¡Modelo publicado exitosamente en: https://huggingface.co/{model_repo_id}")

    # 2. Subir Space
    if not args.skip_space:
        if not space_dir.exists():
            print(f"❌ Error: La carpeta del space no existe en {space_dir}")
            sys.exit(1)

        print(f"\n🎧 [2/2] Creando/Verificando repositorio de Space: {space_repo_id}...")
        api.create_repo(repo_id=space_repo_id, repo_type="space", space_sdk="static", exist_ok=True)

        print(f"🚀 Subiendo aplicación Space (Static HTML/JS) con los 5 ejemplos de prueba...")
        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=space_repo_id,
            repo_type="space",
        )
        print(f"✅ ¡Space publicado exitosamente en: https://huggingface.co/spaces/{space_repo_id}")

    print("\n🎉 ¡Proceso completado con éxito!")
    print(f"👉 Repositorio del Modelo: https://huggingface.co/{model_repo_id}")
    print(f"👉 Demo Interactiva Space: https://huggingface.co/spaces/{space_repo_id}")


if __name__ == "__main__":
    main()

