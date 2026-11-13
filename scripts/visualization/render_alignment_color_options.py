"""Render occlusion color alternatives using the registered Blender camera."""
from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--palette', choices=['coral', 'original', 'modern', 'light'], default='coral')
parser.add_argument('--opaque-background', action='store_true', help='Use an opaque background instead of transparent PNGs')
parser.add_argument('--device', choices=['CPU', 'CUDA', 'OPTIX'], default='CPU')
parser.add_argument('--force', action='store_true', help='Replace existing renders')
parser.add_argument('--base', type=Path, help='Aligned scene folder containing mesh_panorama_aligned.ply')
args = parser.parse_args()
if args.base:
    ALIGNMENT_BASE = args.base.resolve()

setup = Path(__file__).with_name('rerender_alignment_gray.py')
exec(compile(setup.read_text().split('bpy.ops.render.render(write_still=True)')[0], str(setup), 'exec'))
from PIL import Image, ImageDraw, ImageFont

scene.render.film_transparent = not args.opaque_background
scene.render.image_settings.color_mode = 'RGBA' if scene.render.film_transparent else 'RGB'

if args.device != 'CPU':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = args.device
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == args.device]
    if not devices:
        raise RuntimeError(f'No {args.device} GPU available')
    for device in prefs.devices:
        device.use = device.type == args.device
    scene.cycles.device = 'GPU'
    print('Rendering on:', [d.name for d in devices], flush=True)

folder = BASE / 'qualitative_view04_closeups/occlusion_color_options'
if scene.render.film_transparent:
    folder = folder / (args.palette + '_transparent')
elif args.palette != 'original':
    folder = folder / args.palette
folder.mkdir(parents=True, exist_ok=True)
options = {
    'lavender': (166, 123, 219),
    'teal': (48, 183, 164),
    'coral': (235, 114, 132),
    'gold': (237, 184, 62),
    'sky_blue': (88, 158, 235),
    'mint': (135, 213, 146),
    'orange': (242, 143, 61),
    'magenta': (217, 105, 199),
}
if args.palette == 'coral':
    options = {'coral': (235, 114, 132)}
elif args.palette == 'modern':
    options = {
        'cobalt': (42, 86, 210),
        'emerald': (16, 137, 100),
        'burnt_sienna': (196, 77, 43),
        'deep_raspberry': (174, 38, 91),
    }
elif args.palette == 'light':
    options = {
        'clear_azure': (92, 190, 245),
        'bright_jade': (65, 205, 157),
        'apricot': (255, 175, 99),
        'watermelon': (248, 116, 143),
    }
manifest = dict(camera_location=list(cam.location), camera_rotation_euler=list(cam.rotation_euler), ortho_scale=cam.data.ortho_scale, resolution=[1600,1400], samples=scene.cycles.samples, options={})
manifest['device'] = args.device
manifest['transparent_background'] = scene.render.film_transparent
for name, rgb in options.items():
    c[hidden[used], :3] = np.array(rgb, dtype=np.float32) / 255
    attr.data.foreach_set('color_srgb', c.ravel())
    data.update()
    scene.render.filepath = str(folder / (name+'.png'))
    if args.force or not (folder / (name+'.png')).exists():
        bpy.ops.render.render(write_still=True)
    manifest['options'][name] = dict(rgb=rgb, file=name+'.png')
(folder/'render_settings.json').write_text(json.dumps(manifest,indent=2)+'\n')
columns = min(4, len(options))
canvas = Image.new('RGBA',(800 * columns,740 * ((len(options)+3)//4)),(0,0,0,0) if scene.render.film_transparent else (255,255,255,255))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
except OSError:
    font = ImageFont.load_default()
for i,(name,rgb) in enumerate(options.items()):
    x,y=(i%4)*800,(i//4)*740
    with Image.open(folder/(name+'.png')) as im:
        canvas.paste(im.resize((800,700),Image.Resampling.LANCZOS),(x,y+40))
    draw.rectangle((x+20,y+9,x+43,y+32),fill=rgb)
    draw.text((x+55,y+5),name.replace('_', ' ').title(),fill='black',font=font)
canvas.save(folder/'comparison.png')
print(folder/'comparison.png',flush=True)
