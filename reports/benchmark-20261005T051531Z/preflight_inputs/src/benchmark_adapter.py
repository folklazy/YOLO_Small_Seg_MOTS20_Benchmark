"""Explicit common NMS setup; inherits byte-frozen pilot prediction/timing."""
from pathlib import Path
from types import SimpleNamespace
from yolo_adapter import YOLOAdapter
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils.checks import check_imgsz

class AuditedPredictor(SegmentationPredictor):
    def construct_result(self,pred,img,orig_img,img_path,proto):
        self.post_nms_candidates=int(pred.shape[0])
        if self.post_nms_candidates>=1000:
            raise RuntimeError('Model max_det saturated: candidate retention cannot be verified')
        return super().construct_result(pred,img,orig_img,img_path,proto)

class Adapter(YOLOAdapter):
    def __init__(self,checkpoint,config,framework_dir):
        checkpoint=Path(checkpoint)
        assert checkpoint.is_file()
        model=YOLO(str(checkpoint),task='segment')
        assert model.task=='segment' and model.names[0]=='person'
        self.person_class=0
        overrides=dict(task='segment',mode='predict',device=config['device'],batch=1,imgsz=640,
                       conf=.001,iou=.7,max_det=1000,classes=[0],retina_masks=True,rect=False,
                       augment=False,nms=True,agnostic_nms=False,quantize=32,
                       save=False,save_txt=False,verbose=False,visualize=False,
                       project=str(framework_dir),name='predict',exist_ok=False)
        self.predictor=AuditedPredictor(overrides=overrides)
        self.predictor.setup_model(model=model.model,verbose=False)
        self.predictor.imgsz=check_imgsz(640,stride=self.predictor.model.stride,min_dim=2)
        self.predictor.source_type=SimpleNamespace(tensor=False,from_img=True,stream=False)
        self.device=self.predictor.device
        self.precision='fp16' if self.predictor.model.fp16 else 'fp32'
        assert self.device.type=='cuda' and self.precision=='fp32'
        assert self.predictor.model.end2end is False
        self.runtime_parameters=sum(x.numel() for x in self.predictor.model.model.parameters())
