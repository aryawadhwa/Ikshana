import pytesseract

# Point pytesseract to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# You can use pytesseract here to extract text from an image, e.g.:
# print(pytesseract.image_to_string('/path/to/your-image.png'))