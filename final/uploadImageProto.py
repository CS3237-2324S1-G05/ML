from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os

app = Flask(__name__)

# Configure the "uploads" folder to store uploaded images
app.config['PEOPLE_FOLDER'] = 'people'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=["GET", "POST"])
def upload_image():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['PEOPLE_FOLDER'], filename))
        return 'File uploaded successfully'

    return 'File upload failed. Please check the file format.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
