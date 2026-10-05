"""Meaningful metric/ignore regression checks; no inference or dataset writes."""
from pathlib import Path
import unittest
import math
import io
from contextlib import redirect_stdout
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mots import Frame, PersonGT, encode, load_frames, validate_known_gt
from metrics import Prediction, fixed_metrics, ap_metrics

ROOT = Path(__file__).resolve().parents[2]


def boxmask(x, y, w, h, shape=(20, 20)):
    mask = np.zeros(shape, dtype=bool)
    mask[y:y+h, x:x+w] = True
    return mask


def frame(masks, ignore=None):
    return Frame("synthetic", 1, Path("synthetic.jpg"), 20, 20,
                 [PersonGT(i+1, encode(m)) for i, m in enumerate(masks)],
                 encode(np.zeros((20, 20), bool) if ignore is None else ignore),
                 int(ignore is not None))


def pred(mask, score=0.9):
    return Prediction(score, 0, [0, 0, 20, 20], mask)


class PipelineTests(unittest.TestCase):
    def test_lossless_bit_packing_and_padding(self):
        import torch
        from yolo_adapter import pack_binary_masks
        for shape in [(3,7,11),(2,8,16),(0,4,4)]:
            a=np.random.default_rng(41).integers(0,2,size=shape,dtype=np.uint8)
            packed=pack_binary_masks(torch.from_numpy(a)).numpy()
            expected=np.packbits(a.reshape(shape[0],shape[1]*shape[2]),axis=1,bitorder='little')
            np.testing.assert_array_equal(packed,expected)

    def test_compact_metrics_do_not_decode_dense_masks(self):
        from unittest.mock import patch
        from metrics import PackedPrediction
        a=boxmask(1,1,4,4)
        dense=pred(a)
        packed=PackedPrediction(.9,0,[0,0,20,20],np.packbits(a,bitorder='little'),a.shape)
        compact=packed.compact()
        self.assertEqual(compact.rle,dense.rle)
        f=frame([a]);expected=fixed_metrics(f,[dense])
        with patch('pycocotools.mask.decode',side_effect=AssertionError('Unexpected dense decode')):
            self.assertEqual(fixed_metrics(f,[compact]),expected)
            self.assertAlmostEqual(ap_metrics([f],[[compact]])[0]['map50_95'],1.)
        np.testing.assert_array_equal(compact.mask,a)

    def test_diagnostic_cap_keeps_default_ap_semantics(self):
        f=frame([boxmask(1,1,4,4)])
        ps=[pred(boxmask(10,10,4,4),.99-i*.001) for i in range(100)]
        ps.append(pred(boxmask(1,1,4,4),.1))
        self.assertEqual(ap_metrics([f],[ps])[0]['ap50'],0.)
        expanded,_,details=ap_metrics([f],[ps],max_dets=200,return_details=True)
        self.assertAlmostEqual(expanded['ap50'],1/101)
        self.assertEqual(details[0]['tail_tp_ranks'][0],[101])

    def test_summary_preserves_numeric_precision(self):
        from run_smoke import build_summary
        row={'tp':2,'fp':1,'fn':2,'gt_persons':4,'ignored_predictions':1,
             'preprocess_ms':1.,'inference_ms':2.,'postprocess_ms':3.,'total_ms':6.}
        cfg={'sequence':'synthetic','fixed_confidence':.25,'matching_iou':.5,'imgsz':640}
        result=build_summary([row],[{'matches':[{'iou':.5,'dice':2/3}]}],{}, {},cfg,
                             'local.pt','cuda:0','fp32')
        self.assertAlmostEqual(result['precision'],2/3)
        self.assertEqual(result['precision_mode'],'fp32')
        self.assertAlmostEqual(result['fps'],1000/6)

    def test_known_mots_reference(self):
        f = load_frames(ROOT/'datasets/MOTS/MOTS/train', 'MOTS20-02', [1])[0]
        self.assertEqual(validate_known_gt(f)['area'], 28863)
        self.assertGreater(len(f.persons), 1)
        self.assertGreater(f.ignore_count, 0)

    def test_perfect_multiple_persons_and_ap(self):
        a, b = boxmask(1, 1, 4, 4), boxmask(10, 10, 4, 4)
        f, ps = frame([a, b]), [pred(a), pred(b, .8)]
        r = fixed_metrics(f, ps)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (2, 0, 0))
        for key in ['precision','recall','f1','matched_mask_iou','matched_dice']:
            self.assertEqual(r[key], 1)
        ap, _ = ap_metrics([f], [ps])
        for key in ['ap50','ap75','map50_95']:
            self.assertAlmostEqual(ap[key], 1)

    def test_duplicate_detection_one_to_one(self):
        a = boxmask(1, 1, 4, 4)
        r = fixed_metrics(frame([a]), [pred(a), pred(a)])
        self.assertEqual((r['tp'],r['fp'],r['fn']), (1,1,0))
        self.assertEqual(r['matches'][0]['prediction_index'], 0)
        self.assertAlmostEqual(r['precision'], .5)
        self.assertAlmostEqual(r['f1'], 2/3)

    def test_empty_predictions_and_ap(self):
        f = frame([boxmask(1, 1, 4, 4)])
        r = fixed_metrics(f, [])
        self.assertEqual((r['tp'],r['fp'],r['fn']), (0,0,1))
        self.assertTrue(math.isnan(r['precision']))
        self.assertEqual(r['recall'], 0)
        self.assertEqual(r['f1'], 0)
        self.assertTrue(math.isnan(r['matched_mask_iou']))
        ap, _ = ap_metrics([f], [[]])
        self.assertEqual(ap['map50_95'], 0)

    def test_empty_gt_is_not_perfect(self):
        f = frame([])
        r = fixed_metrics(f, [pred(boxmask(1, 1, 4, 4))])
        self.assertEqual((r['tp'],r['fp'],r['fn']), (0,1,0))
        self.assertEqual(r['precision'], 0)
        self.assertTrue(math.isnan(r['recall']))
        ap, _ = ap_metrics([f], [[]])
        self.assertTrue(math.isnan(ap['ap50']))

    def test_hand_computed_iou_dice_thresholds(self):
        a, b = boxmask(1, 1, 3, 2), boxmask(2, 1, 3, 2)
        f = frame([a])
        r = fixed_metrics(f, [pred(b)], iou_threshold=.5)
        self.assertEqual(r['tp'], 1)
        self.assertAlmostEqual(r['matched_mask_iou'], .5)
        self.assertAlmostEqual(r['matched_dice'], 2/3)
        self.assertEqual(fixed_metrics(f, [pred(b)], iou_threshold=.51)['fn'], 1)
        ap, _ = ap_metrics([f], [[pred(b)]])
        self.assertAlmostEqual(ap['ap50'], 1)
        self.assertEqual(ap['ap75'], 0)
        self.assertAlmostEqual(ap['map50_95'], .1)

    def test_ignore_ioa_not_iou_and_valid_priority(self):
        a = boxmask(1, 1, 3, 3)
        ignored = boxmask(10, 0, 10, 20)
        f = frame([a], ignored | a)
        ps = [pred(boxmask(12, 2, 2, 2), .99), pred(a, .9)]
        r = fixed_metrics(f, ps)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['ignored_predictions']), (1,0,0,1))
        ap, _ = ap_metrics([f], [ps])
        self.assertAlmostEqual(ap['map50_95'], 1)
        self.assertEqual(ap['ignored_detections_at_ap50'], 1)
        without, _ = ap_metrics([f], [ps], use_ignore=False)
        self.assertLess(without['ap50'], ap['ap50'])

    def test_partial_ignore_constant_across_ap_thresholds(self):
        a = boxmask(1, 1, 2, 2)
        # Prediction is 60% inside ignored region: suppress even at AP95.
        f = frame([a], boxmask(10, 10, 3, 2))
        ps = [pred(boxmask(10, 10, 5, 2), .99), pred(a, .9)]
        self.assertEqual(fixed_metrics(f, ps)['ignored_predictions'], 1)
        ap, _ = ap_metrics([f], [ps])
        self.assertAlmostEqual(ap['map50_95'], 1)
        ps[0] = pred(boxmask(10, 10, 8, 2), .99)
        self.assertEqual(fixed_metrics(f, ps)['fp'], 1)

    def test_low_confidence_kept_only_for_ap(self):
        a = boxmask(1, 1, 4, 4)
        f, ps = frame([a]), [pred(a, .1)]
        self.assertEqual(fixed_metrics(f, ps, confidence=.25)['fn'], 1)
        self.assertAlmostEqual(ap_metrics([f], [ps])[0]['ap50'], 1)

    def test_shape_mismatch_rejected(self):
        f = frame([boxmask(1,1,4,4)])
        ps = [pred(np.ones((10,10),bool))]
        with self.assertRaises(ValueError): fixed_metrics(f, ps)
        with self.assertRaises(ValueError): ap_metrics([f], [ps])

    def test_ap_matches_unmodified_coco_without_ignore(self):
        a, b = boxmask(1,1,4,4), boxmask(10,10,4,4)
        f = frame([a,b])
        ps = [pred(boxmask(0,15,2,2),.99), pred(a,.9), pred(b,.8)]
        custom, _ = ap_metrics([f], [ps])
        with redirect_stdout(io.StringIO()):
            gt=COCO()
            gt.dataset={'info':{},'images':[{'id':1,'height':20,'width':20}],
                        'categories':[{'id':1,'name':'Person'}],
                        'annotations':[{'id':i+1,'image_id':1,'category_id':1,
                                        'segmentation':encode(m),'area':16,'bbox':bb,'iscrowd':0}
                                       for i,(m,bb) in enumerate([(a,[1,1,4,4]),(b,[10,10,4,4])])]}
            gt.createIndex()
            dt=gt.loadRes([{'image_id':1,'category_id':1,'segmentation':p.rle,'score':p.confidence} for p in ps])
            e=COCOeval(gt,dt,'segm');e.evaluate();e.accumulate();e.summarize()
        self.assertAlmostEqual(custom['ap50'], e.stats[1])
        self.assertAlmostEqual(custom['ap75'], e.stats[2])
        self.assertAlmostEqual(custom['map50_95'], e.stats[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
