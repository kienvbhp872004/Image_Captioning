import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from models.caption_model import CaptionModel
from config import Config
from main import create_dataloader
from datasets.transforms import get_transforms


class ImageCaptioningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning App")
        self.root.geometry("600x600")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beam_size = 3

        # ===== Load model =====
        self.cfg = Config()
        self.cfg.device = self.device

        _, self.vocab = create_dataloader(self.cfg)
        self.cfg.vocab_size = len(self.vocab)

        self.model = CaptionModel(
            self.cfg.vocab_size,
            self.cfg.embed_size,
            self.cfg.num_heads,
            self.cfg.num_layers
        ).to(self.device)

        state = torch.load(
            "saved/model/Model_epoch_9.pth",
            map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.eval()

        self.transforms = get_transforms()

        # ===== UI =====
        self.image_path = None

        self.btn_upload = tk.Button(
            root, text="📂 Upload Image", command=self.upload_image, width=20
        )
        self.btn_upload.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.btn_generate = tk.Button(
            root, text="📝 Generate Caption", command=self.generate_caption, width=20
        )
        self.btn_generate.pack(pady=10)

        self.caption_text = tk.Text(
            root, height=4, width=60, wrap="word", font=("Arial", 12)
        )
        self.caption_text.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if not file_path:
            return

        self.image_path = file_path
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((350, 350))

        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)

        self.caption_text.delete(1.0, tk.END)

    def generate_caption(self):
        if self.image_path is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        image = Image.open(self.image_path).convert("RGB")
        image = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.model.generate_caption_beamsearch(
                image=image,
                vocab=self.vocab,
                beam_size=self.beam_size,
                device=self.device
            )

        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, caption)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningGUI(root)
    root.mainloop()
