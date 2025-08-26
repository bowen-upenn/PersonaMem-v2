Please download image data from Google Drive at https://drive.google.com/file/d/1TnKOvKDgxf7yxR7MlNCw6IzZjObnSHTG/view?usp=sharing and put them under the path data/photobook_images.

Reference: https://dmg-photobook.github.io/datasets.html.

<!-- 
Script to download the entire Google Drive folder:

```bash
# Install gdown if not already installed
pip install gdown

# Download the entire folder
gdown --folder https://drive.google.com/file/d/1TnKOvKDgxf7yxR7MlNCw6IzZjObnSHTG/view?usp=sharing -O data

# Unzip the folder
unzip data/photobook_images.zip -d data/photobook_images/
