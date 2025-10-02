import os
from dotenv import load_dotenv

load_dotenv()

from app.frontend.gradio_app import build_interface


def main() -> None:
    iface = build_interface()
    # For HF Spaces, server_name defaults work; expose 7860
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()




