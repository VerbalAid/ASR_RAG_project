import qrcode

def generate_qr(url, output_file="qr_code.png"):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=4,
    )
    
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_file)

    print(f"QR code saved as {output_file}")

if __name__ == "__main__":
    url = "https://github.com/VerbalAid/ASR_RAG_project/"
    output_file = "ASR_RAG_project_qr.png"

    generate_qr(url, output_file)