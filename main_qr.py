import qrcode

# Data to encode
data = "https://mieuxvoter.fr"

# Create QR code instance
qr = qrcode.QRCode(
    version=1,  # controls the size of the QR code (1-40)
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # about 7% or less error correction
    box_size=10,  # size of each box in pixels
    border=4,  # thickness of the border (boxes)
)

# Add data to the instance
qr.add_data(data)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill="black", back_color="white")

# Save the image
img.save("qrcode_mieux_voter.png")
