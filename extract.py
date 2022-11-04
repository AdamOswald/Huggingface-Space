from colorthief import ColorThief
from pathlib import Path
import json
from PIL import Image


images_path = Path('frontend/static/images')
images = images_path.glob("*.[jpeg jpg png]*")
print(images)
data = {}
for image in images:
    print(image.stem)
    image_pil = Image.open(image)
    color_thief = ColorThief(image)
    image_pil.save(Path.joinpath(images_path, (image.stem + ".jpg")), optimize=True, quality=95)
    prompt = image.stem.split("-")[2]
    try:
        type(data[prompt]) == list
    except:
        data[prompt] = []

    colors = color_thief.get_palette(color_count=5, quality=1)
    colors_hex = ['#%02x%02x%02x' % (color) for color in colors]
    data[prompt].append({
        "colors": colors_hex,
        "imgURL": "static/images/" + image.stem + ".jpg"
    })
prompts = [{"prompt": prompt, "images": values}
           for (prompt, values) in data.items()]
with open('frontend/static/data.json', 'w') as f:
    json.dump(prompts, f)
