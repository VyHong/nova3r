"""Render a saved mesh view with several Taubin smoothing strengths."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def sheet(items, output, crop=False):
    width = 900
    tiles = []
    for count, path in items:
        image = Image.open(path).convert('RGB')
        if crop:
            w, h = image.size
            image = image.crop((int(w*.18), int(h*.28), int(w*.52), int(h*.68)))
        height = round(image.height * width / image.width)
        tile = Image.new('RGB', (width, height+46), '#eeeeee')
        tile.paste(image.resize((width,height), Image.Resampling.LANCZOS), (0,46))
        try: font = ImageFont.truetype('DejaVuSans.ttf', 25)
        except OSError: font = ImageFont.load_default()
        label = f'{count} iterations' + (' (original smoothing level)' if count == 30 else '')
        ImageDraw.Draw(tile).text((16,9),label,fill='#111111',font=font)
        tiles.append(tile)
    canvas = Image.new('RGB',(3*width, ((len(tiles)+2)//3)*tiles[0].height),'white')
    for i,tile in enumerate(tiles): canvas.paste(tile,((i%3)*width,(i//3)*tile.height))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--blender',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[0,5,15,30,60,120])
    args = parser.parse_args()
    source_image = args.image.resolve()
    metadata = json.loads(source_image.with_suffix('.settings.json').read_text())
    view = Path(metadata['view_source'])
    if not view.is_absolute(): view = ROOT/view
    source_mesh = source_image.parent/'test_mesh.ply'
    out = source_image.parent/'smoothing_levels'
    out.mkdir(exist_ok=True)
    if view.exists():
        shutil.copyfile(view,out/'view.json')
    else:
        # The artifact folder was moved without its Open3D camera JSON. Its
        # render sidecar retains pose, focal length and image dimensions.
        w,h = metadata['resolution']
        focal = metadata['lens_mm']*w/36
        reconstructed = dict(width=w,height=h,
            intrinsic_matrix=[[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1]],
            blender_camera_matrix_world=metadata['camera_matrix_world'],
            mesh_matrix_world=[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],
            recovered_from=str(source_image.with_suffix('.settings.json')),
            assumptions='Standard Open3D square pixels, centered principal point and NOVA3R Y-up conversion')
        (out/'view.json').write_text(json.dumps(reconstructed,indent=2)+'\n')
    items = []
    for iterations in args.levels:
        if iterations < 0: parser.error('Smoothing levels must be nonnegative')
        print(f'Smoothing level {iterations}',flush=True)
        mesh = source_mesh
        if iterations:
            existing = source_image.parent/'test_mesh_render_smoothed.ply'
            if iterations == metadata['preprocessing']['iterations'] and existing.exists():
                mesh = existing
            else:
                mesh = out/f'mesh_taubin_{iterations:03d}.ply'
                subprocess.run([sys.executable,str(ROOT/'scripts/visualization/preprocess_render_mesh.py'),
                                str(source_mesh),'--output',str(mesh),'--iterations',str(iterations)],check=True)
        output = out/f'taubin_{iterations:03d}.png'
        argv = ['render_saved_mesh_view.py',str(out/'view.json'),'--mesh',str(mesh),
                '--output',str(output),'--exposure',str(metadata.get('exposure',-.4))]
        for key,flag in [('transparent_background','--transparent'),('soft_lighting','--soft-lighting'),('multi_fill','--multi-fill')]:
            if metadata.get(key): argv.append(flag)
        expr = 'import sys,runpy; sys.argv='+repr(argv)+'; runpy.run_path('+repr(str(ROOT/'scripts/visualization/render_saved_mesh_view.py'))+',run_name="__main__")'
        with (out/f'taubin_{iterations:03d}.render.log').open('w') as log:
            subprocess.run([str(args.blender),'-b','--python-expr',expr],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        if not output.exists(): raise RuntimeError(f'Missing render: {output}')
        items.append((iterations,output))
        (out/'manifest.json').write_text(json.dumps(dict(source=str(source_image),source_mesh=str(source_mesh),
              method='Taubin',lambda_filter=.5,mu=-.53,source_modified=False,
              renders=[dict(iterations=n,image=str(p)) for n,p in items]),indent=2)+'\n')
        print(f'Rendered {output}',flush=True)
    sheet(items,out/'comparison.png')
    sheet(items,out/'comparison_closeup.png',crop=True)
    print(f'Finished: {out}',flush=True)


if __name__ == '__main__': main()
