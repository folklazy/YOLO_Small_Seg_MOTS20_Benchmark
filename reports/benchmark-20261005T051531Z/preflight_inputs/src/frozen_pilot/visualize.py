"""Small debugging contact sheets, called only outside measured inference."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask


def save_debug(frame, bgr, predictions, fixed, output, confidence):
    rgb = bgr[:, :, ::-1].copy()
    font = ImageFont.load_default(size=16)
    image_font = ImageFont.load_default(size=40)
    palette = [(50,220,90),(40,170,255),(255,210,50),(220,100,255),(255,140,60),(70,240,220)]
    gt_masks = [g.decode() for g in frame.persons]
    def panel(title, items):
        pixels = rgb.copy()
        for mask, color, _ in items:
            pixels[mask] = (.5*pixels[mask]+.5*np.array(color)).astype(np.uint8)
        im = Image.fromarray(pixels)
        draw = ImageDraw.Draw(im)
        for mask, color, label in items:
            yy, xx = np.nonzero(mask)
            if len(xx):
                draw.text((int(xx.mean()),int(yy.mean())),label,fill=color,
                          font=image_font,stroke_width=3,stroke_fill='black')
        im.thumbnail((640,360))
        out = Image.new('RGB',(640,400),'#171717')
        out.paste(im,(0,40))
        ImageDraw.Draw(out).text((8,8),title,fill='white',font=font)
        return out
    panels = [panel(f'{frame.sequence} / {frame.number:06d}: source',[])]
    panels.append(panel('GT people: labels are object IDs',[(m,palette[i%len(palette)],str(g.object_id))
                        for i,(g,m) in enumerate(zip(frame.persons,gt_masks))]))
    panels.append(panel(f'Predictions: confidence >= {confidence}',[(p.mask,palette[i%len(palette)],f'P{i} {p.confidence:.2f}')
                        for i,p in enumerate(predictions) if p.confidence>=confidence]))
    panels.append(panel('Matched predictions: P index -> GT ID',[(predictions[m['prediction_index']].mask,
                        palette[m['gt_index']%len(palette)],f"P{m['prediction_index']}->{m['object_id']}")
                        for m in fixed['matches']]))
    unmatched = [(gt_masks[i],(255,170,30),f'FN {frame.persons[i].object_id}') for i in fixed['fn_indices']]
    unmatched += [(predictions[i].mask,(255,50,50),f'FP P{i}') for i in fixed['fp_indices']]
    panels.append(panel('Unmatched: orange GT (FN), red predictions (FP)',unmatched))
    ignored = [(coco_mask.decode(frame.ignore_rle).astype(bool),(220,60,230),'IGNORE')]
    ignored += [(predictions[i].mask,(150,150,150),f'ignored P{i}') for i in fixed['ignored_indices']]
    panels.append(panel('Magenta: ignore region; gray: ignored predictions',ignored))
    sheet = Image.new('RGB',(1920,800))
    for i,p in enumerate(panels): sheet.paste(p,((i%3)*640,(i//3)*400))
    sheet.save(output)
